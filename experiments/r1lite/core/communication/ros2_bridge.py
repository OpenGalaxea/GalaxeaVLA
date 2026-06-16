
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
from functools import partial
from loguru import logger
import numpy as np
import time
from dataclasses import asdict

from core.communication.robot_topics import RobotTopicsConfig
from core.communication.message_queue import MessageQueue

from utils.message.message_convert import (
    header_stamp_to_timestamp,
    pose_to_7d_array,
    compressed_image_to_rgb_array,
    array_to_joint_state
)
from utils.message.datatype import RobotAction

class Ros2Bridge:
    def __init__(self, config, use_recv_time: bool = False, num_threads: int = None):
        """
        Initialize the ROS2 bridge.

        Args:
            config: configuration dictionary loaded from config.toml.
            use_recv_time: whether to use receive time.
            num_threads: executor thread count. None means auto-select, usually
                based on CPU core count.
        """
        self.topics_config = RobotTopicsConfig()
        self.use_recv_time = use_recv_time
        
        # Initialize ROS2.
        if not rclpy.ok():
            rclpy.init()
        
        self.node = rclpy.create_node("ros2_bridge")
        self.subscribers = {}
        self.publishers = {}
        self.dof_of_arm = 6#cfg.data.shape_meta["state"][0]["raw_shape"] # left arm meta
        self.obs_buffer = {}
        
        # Create a callback group that allows concurrent callbacks.
        self.callback_group = ReentrantCallbackGroup()
        
        # Create the multi-threaded executor.
        self.executor = MultiThreadedExecutor(num_threads=num_threads)
        
        self.init_topics()

        self.executor.add_node(self.node)
        
        # Start the executor in a separate thread.
        import threading
        self._executor_thread = threading.Thread(target=self._run_executor, daemon=True)
        self._executor_thread.start()
        logger.info(f"Multi-threaded executor started with callback groups")

    def init_topics(self):
        """Initialize subscriptions and publishers with callback-group threading."""
        for name, topic in self.topics_config.images.items():
            self.obs_buffer[name] = MessageQueue(maxlen=self.topics_config.camera_deque_length)
            self.subscribers[topic] = self.node.create_subscription(
                self.topics_config.message_type["images"], 
                topic, 
                partial(self.image_callback, _stack=self.obs_buffer[name]), 
                self.topics_config.qos["sub"],
                callback_group=self.callback_group  # Use callback group.
            )

        for name, topic in self.topics_config.state.items():
            self.obs_buffer[name] = MessageQueue(maxlen=self.topics_config.state_deque_length)
            logger.info(f'{name}: {topic}')
            if "ee_pose" in name:
                self.subscribers[topic] = self.node.create_subscription(
                    self.topics_config.message_type["pose"], 
                    topic, 
                    partial(self.pose_callback, _stack=self.obs_buffer[name]), 
                    self.topics_config.qos["sub"],
                    callback_group=self.callback_group  # Use callback group.
                )
            else:
                self.subscribers[topic] = self.node.create_subscription(
                    self.topics_config.message_type["state"], 
                    topic, 
                    partial(self.state_callback, _stack=self.obs_buffer[name], state_name=name), 
                    self.topics_config.qos["sub"],
                    callback_group=self.callback_group  # Use callback group.
                )

        for name, topic in self.topics_config.action.items():
            self.publishers[topic] = self.node.create_publisher(
                self.topics_config.message_type["action"], topic, self.topics_config.qos["pub"])

    def is_running(self):
        return rclpy.ok()

    def publish_action(self, action: RobotAction):
        for name, msg in asdict(action).items():
            if msg is not None:
                self.publishers[self.topics_config.action[name]].publish(msg)

    def reset(self, step_size=0.2, freq = 5):
        # reset to zero joints and close grippers
        print("resetting")
        while True:
            joint_feedback_topics = [topic for topic in self.topics_config.state.keys() if 'ee' not in topic]
            action = {}
            for name in joint_feedback_topics:
                if "gripper" in name:
                    # close grippers
                    # action[name] = np.array([0])
                    action[name] = [0.]
                elif "chassis" in name or "torso" in name:
                    continue
                else:
                    latest_feedback = np.array(self.obs_buffer[name][-1]['data'][:6])
                    step_size_with_dir = step_size * np.sign(-latest_feedback)
                    step_size_with_dir = np.where(np.abs(latest_feedback) < step_size,
                                                  -latest_feedback, step_size_with_dir)
                    motion_target = latest_feedback + step_size_with_dir
                    action[name] = motion_target.tolist()

            for k, v in action.items():
                msg = array_to_joint_state(v)
                self.publishers[self.topics_config.action[k]].publish(msg)
            if all([np.all(np.array(value) == 0) for _, value in action.items()]):
                break
            time.sleep(1 / freq)

    def register_publish_callback(self, frequency: float, callback: callable):
        self.timer = self.node.create_timer(1.0 / frequency, callback)

    def register_subscription(self, message_type: type, topic: str, callback: callable):
        self.subscribers[topic] = self.node.create_subscription(
            message_type,
            topic,
            callback,
            self.topics_config.qos["sub"],
            callback_group=self.callback_group
        )


    def _find_nearest_message(self, buffer: MessageQueue, target_time: float) -> dict:
        """
        Find the message in buffer closest to target_time.
        Optimization: search backward because the buffer is time-ordered and the
        newest data is at the end.
        """
        if len(buffer) == 0:
            return None
        
        best_msg = None
        best_diff = float('inf')
        
        # Search backward; this is more likely to find a close match.
        for i in range(len(buffer) - 1, -1, -1):
            msg = buffer[i]
            time_diff = abs(msg["message_time"] - target_time)
            if time_diff < best_diff:
                best_diff = time_diff
                best_msg = msg
                # Exit early when the match is close enough, within 1 ms.
                if time_diff < 0.001:
                    break
        
        return best_msg
    
    def gather_obs(self):
        """Gather raw observations for the remote inference server.

        Returns a dict matching the server protocol::

            {
                "images": {name: ndarray [C,H,W] uint8, ...},
                "state":  {name: ndarray [D] float32, ...},
            }

        Returns None if any required buffer is empty.
        """
        head_rgb_key = "head_rgb"
        if head_rgb_key not in self.obs_buffer or len(self.obs_buffer[head_rgb_key]) == 0:
            logger.warning("Head camera buffer is empty")
            return None

        head_buffer = self.obs_buffer[head_rgb_key]
        head_msg = head_buffer[-1]
        reference_time = head_msg["message_time"]

        obs = {"images": {}, "state": {}}

        for name, buffer in self.obs_buffer.items():
            if len(buffer) == 0:
                logger.warning(f"Buffer {name} is empty, skipping")
                return None

            if name == head_rgb_key:
                data = head_msg["data"]
            else:
                nearest_msg = self._find_nearest_message(buffer, reference_time)
                if nearest_msg is None:
                    logger.warning(f"Failed to find nearest message for {name}")
                    return None
                data = nearest_msg["data"]

            if not isinstance(data, np.ndarray):
                data = np.asarray(data)

            if name in self.topics_config.images:
                # Already (C, H, W) uint8 from compressed_image_to_rgb_array
                obs["images"][name] = data
            else:
                obs["state"][name] = data.astype(np.float32)

            if name == "chassis":
                obs["state"][name] = obs["state"][name][:3]
                obs["state"][name] = np.arctan2(
                    np.sin(obs["state"][name]),
                    np.cos(obs["state"][name]),
                ).astype(np.float32)

        return obs


    def _run_executor(self):
        """Run the executor in its own thread."""
        try:
            self.executor.spin()
        except Exception as e:
            logger.error(f"Executor error: {e}")
        finally:
            logger.info("Executor stopped")
    
    def now(self):
        """Return the current time in seconds."""
        return self.node.get_clock().now().nanoseconds * 1e-9
    
    def destroy(self):
        """Clean up resources."""
        # Stop the executor.
        self.executor.shutdown()
        
        # Wait for the executor thread to exit.
        if self._executor_thread and self._executor_thread.is_alive():
            self._executor_thread.join(timeout=2.0)
            if self._executor_thread.is_alive():
                logger.warning("Executor thread did not stop in time")
        
        # Clean up subscribers and publishers.
        for subscriber in self.subscribers.values():
            subscriber.destroy()
        for publisher in self.publishers.values():
            publisher.destroy()
        
        # Remove the node.
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        
        # Shut down ROS2 if this is the last node.
        if rclpy.ok():
            rclpy.shutdown()
        
        logger.info("ROS2 bridge destroyed")

    def _create_data_dict(self, timestamp, data):
        if self.use_recv_time:
            return {
                "message_time": self.now(),
                "data": data,
                "receive_time": self.now(),
                "header_time": timestamp,
            }
        else:
            return {
                "message_time": timestamp,
                "data": data,
                "receive_time": self.now(),
                "header_time": timestamp,
            }
    
    def image_callback(self, msg: CompressedImage, _stack=None):
        data_dict = self._create_data_dict(
            timestamp=header_stamp_to_timestamp(msg.header.stamp),
            data=compressed_image_to_rgb_array(msg.data))
        _stack.append(data_dict)

    def state_callback(self, msg: JointState, _stack=None, state_name: str="None"):
        if state_name == "chassis":
            data_dict = self._create_data_dict(
                timestamp=header_stamp_to_timestamp(msg.header.stamp),
                data=np.array(msg.position))
            _stack.append(data_dict)
        elif state_name == "torso":
            data_dict = self._create_data_dict(
                timestamp=header_stamp_to_timestamp(msg.header.stamp),
                data=np.array(msg.position))
            _stack.append(data_dict)
        elif "arm" in state_name:
            arm_dims = 6#self.shape_meta["state"][state_name]["raw_shape"]
            data_dict = self._create_data_dict(
                timestamp=header_stamp_to_timestamp(msg.header.stamp),
                data=np.array(msg.position[:arm_dims]))
            _stack.append(data_dict)
        elif "gripper" in state_name:
            data_dict = self._create_data_dict(
                timestamp=header_stamp_to_timestamp(msg.header.stamp),
                data=np.array(msg.position))
            _stack.append(data_dict)
        else:
            raise ValueError(f"Invalid state name: {state_name}")

    def pose_callback(self, msg: PoseStamped, _stack=None):
        data_dict = self._create_data_dict(
            timestamp=header_stamp_to_timestamp(msg.header.stamp),
            data=pose_to_7d_array(msg.pose))
    
        _stack.append(data_dict)
