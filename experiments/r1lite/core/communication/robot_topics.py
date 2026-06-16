"""Robot topic configuration module."""
from typing import Dict, Literal
from dataclasses import dataclass, field
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped

@dataclass
class RobotTopicsConfig:
    state: Dict[str, str] = field(
        default_factory=lambda: {
            "left_arm": "/hdas/feedback_arm_left",
            "right_arm": "/hdas/feedback_arm_right",
            "torso": "/hdas/feedback_torso",
            "chassis": "/hdas/feedback_chassis",
            "left_ee_pose": "/motion_control/pose_ee_arm_left",
            "right_ee_pose": "/motion_control/pose_ee_arm_right",
            "left_gripper": "/hdas/feedback_gripper_left",
            "right_gripper": "/hdas/feedback_gripper_right"
        }
    )

    images: Dict[str, str] = field(
        default_factory=lambda: {
            "head_rgb": "/hdas/camera_head/left_raw/image_raw_color/compressed",
            "left_wrist_rgb": "/hdas/camera_wrist_left/color/image_raw/compressed",
            "right_wrist_rgb": "/hdas/camera_wrist_right/color/image_raw/compressed",
            
        }
    )


    action: Dict[str, str] = field(
        default_factory=lambda: {
            "left_arm": "/motion_target/target_joint_state_arm_left",
            "right_arm": "/motion_target/target_joint_state_arm_right",
            "torso": "/motion_target/target_speed_torso",
            "left_gripper": "/motion_target/target_position_gripper_left",
            "right_gripper": "/motion_target/target_position_gripper_right",
            "left_ee_pose": "/motion_target/target_pose_arm_left",
            "right_ee_pose": "/motion_target/target_pose_arm_right"
        }
    )

    qos: Dict[str, QoSProfile] = field(
        default_factory=lambda: {
            "sub": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE
            ),
            "pub": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE
            ),
        }
    )

    message_type: Dict[str, type] = field(
        default_factory=lambda: {
            "state": JointState,
            "images": CompressedImage,
            "action": JointState,
            "pose": PoseStamped,
        }
    )

    camera_deque_length: int = 3 # 15Hz, for 0.2s
    state_deque_length: int = 80 # >400 Hz, for 0.2s
