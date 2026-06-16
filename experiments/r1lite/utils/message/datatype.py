from dataclasses import dataclass, field
from typing import Optional
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from collections import deque
import time

@dataclass
class RobotAction:
    left_arm: Optional[JointState] = None
    right_arm: Optional[JointState] = None
    torso: Optional[JointState] = None
    left_gripper: Optional[JointState] = None
    right_gripper: Optional[JointState] = None
    chassis: Optional[JointState] = None
    left_ee_pose: Optional[PoseStamped] = None
    right_ee_pose: Optional[PoseStamped] = None

@dataclass
class Trajectory:
    timestamp: float = field(default_factory=lambda: time.time())
    actions: deque = field(default_factory=lambda: deque(maxlen=100))
