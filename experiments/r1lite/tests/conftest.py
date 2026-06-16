import sys
from types import ModuleType
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on sys.path so that `import core`, `import utils`, etc. work.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Stub out `websockets` if not installed, so importing core.inference doesn't fail.
if "websockets" not in sys.modules:
    _ws = ModuleType("websockets")
    _ws_sync = ModuleType("websockets.sync")
    _ws_client = ModuleType("websockets.sync.client")
    _ws_client.ClientConnection = MagicMock
    _ws_client.connect = MagicMock
    _ws.sync = _ws_sync
    _ws_sync.client = _ws_client
    sys.modules["websockets"] = _ws
    sys.modules["websockets.sync"] = _ws_sync
    sys.modules["websockets.sync.client"] = _ws_client

# Stub out ROS2 packages if not installed.
for _mod_name in ("std_msgs", "std_msgs.msg", "sensor_msgs", "sensor_msgs.msg",
                   "geometry_msgs", "geometry_msgs.msg", "builtin_interfaces",
                   "builtin_interfaces.msg", "rclpy", "rclpy.qos", "rclpy.node",
                   "rclpy.executors"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
