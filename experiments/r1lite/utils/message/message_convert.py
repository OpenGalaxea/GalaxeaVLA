import base64
import time

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
from utils.message.datatype import RobotAction, Trajectory

def header_stamp_to_timestamp(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

def timestamp_to_header_stamp(timestamp: float):
    stamp = Time()
    stamp.sec = int(timestamp)
    stamp.nanosec = int((timestamp - stamp.sec) * 1_000_000_000)
    return stamp

def pose_to_7d_array(pose: Pose):
    return np.array([
        pose.position.x, 
        pose.position.y, 
        pose.position.z, 
        pose.orientation.x, 
        pose.orientation.y, 
        pose.orientation.z, 
        pose.orientation.w
    ], dtype=np.float32)

def compressed_image_to_rgb_array(buffer: np.uint8):
    rgb_np = np.frombuffer(buffer, np.uint8)
    # Read as three channels and convert to RGB to match training.
    rgb_bgr = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        return
    res = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    return res.transpose(2, 0, 1)

def array_to_joint_state(array: np.ndarray, timestamp: float = None):
    joint_state = JointState()
    if timestamp is None:
        timestamp = time.time()
    joint_state.header.stamp = timestamp_to_header_stamp(timestamp)
    array = np.atleast_1d(array)
    for data in array:
        joint_state.position.append(data)
    return joint_state

def array_to_pose_stamped(array: np.ndarray, timestamp: float = None):
    """
    Convert a 7D array [x, y, z, qx, qy, qz, qw] to a PoseStamped message.
    """
    pose_stamped = PoseStamped()
    if timestamp is None:
        timestamp = time.time()
    pose_stamped.header.stamp = timestamp_to_header_stamp(timestamp)
    
    if len(array) >= 3:
        pose_stamped.pose.position.x = float(array[0])
        pose_stamped.pose.position.y = float(array[1])
        pose_stamped.pose.position.z = float(array[2])
    
    if len(array) >= 7:
        pose_stamped.pose.orientation.x = float(array[3])
        pose_stamped.pose.orientation.y = float(array[4])
        pose_stamped.pose.orientation.z = float(array[5])
        pose_stamped.pose.orientation.w = float(array[6])
    
    return pose_stamped

def decode_img_from_base64(img_base64: str, output_format="rgb") -> np.ndarray:
    img_data = base64.b64decode(img_base64)
    # Convert binary data to a numpy array.
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    # Decode it back to an image with cv2.imdecode.
    img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if output_format == "rgb":
        return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        return img_array

def action_dict_to_robot_action(actions: dict, timestamp: float = None) -> RobotAction:
    """Convert a single-step server action dict to RobotAction.

    Server returns ``{part_name: ndarray[D]}`` (one step per request).

    Args:
        actions: server response with per-part 1D arrays.
        timestamp: optional ROS timestamp; defaults to ``time.time()``.
    """
    if timestamp is None:
        timestamp = time.time()

    field_configs = [
        ("left_arm", array_to_joint_state),
        ("right_arm", array_to_joint_state),
        ("torso", array_to_joint_state),
        ("left_gripper", array_to_joint_state),
        ("right_gripper", array_to_joint_state),
        ("chassis", array_to_joint_state),
        ("left_ee_pose", array_to_pose_stamped),
        ("right_ee_pose", array_to_pose_stamped),
    ]
    kwargs = {}
    for key, converter in field_configs:
        if key in actions:
            kwargs[key] = converter(actions[key], timestamp)
        else:
            kwargs[key] = None
    return RobotAction(**kwargs)


def actions_dict_to_trajectory(actions: dict, time_step: float=0.0666, num_of_steps: int=32, timestamp: float=None) -> Trajectory:
    """Convert server action dict {part: ndarray [T, D]} to Trajectory."""
    field_configs = [
        ("left_arm", array_to_joint_state, False),
        ("right_arm", array_to_joint_state, False),
        ("torso", array_to_joint_state, False),
        ("left_gripper", array_to_joint_state, False),
        ("right_gripper", array_to_joint_state, False),
        ("chassis", array_to_joint_state, False),
        ("left_ee_pose", array_to_pose_stamped, True),
        ("right_ee_pose", array_to_pose_stamped, True),
    ]

    field_data = {}
    for key, converter, _ in field_configs:
        if key in actions:
            field_data[key] = actions[key]

    trajectory = Trajectory()
    trajectory.timestamp = timestamp if timestamp is not None else time.time()
    
    for i in range(num_of_steps):
        action_timestamp = trajectory.timestamp + i * time_step
        action_kwargs = {
        }
        
        for key, converter, _ in field_configs:
            if key in field_data:
                action_kwargs[key] = converter(field_data[key][i], action_timestamp)
            else:
                action_kwargs[key] = None
        
        action = RobotAction(**action_kwargs)
        trajectory.actions.append(action)
    
    return trajectory
