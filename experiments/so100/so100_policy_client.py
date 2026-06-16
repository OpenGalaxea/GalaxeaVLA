"""SO100 policy client for the G05 serve_policy.py WebSocket server.

Architecture:
- FollowerArm: single background thread owns the serial bus exclusively
  (ported from molmoact2-so101). Producer/consumer never touch the bus.
- CameraWorker: per-camera background thread for non-blocking frame reads.
- InferenceProducer: async loop at action_fps Hz.
  - If need_obs=True: send full obs (images+state) → server infers → get step
  - If need_obs=False: send {} → server returns next cached step
  - Directly calls follower.set_target() with each step's arm-frame action.
  - No ring buffer, no temporal ensembling, no drain loop.
- Main thread: cv2 camera display until Ctrl-C.

Server (ChunkedPolicyWrapper) manages the 16-step chunk cache internally.
Client just ticks at action_fps Hz and follows need_obs flag.

Frame convention (lerobot v3.0 ↔ training v2.1):
  arm  → model : signs * arm  + offsets
  model → arm  : (model - offsets) * signs
  signs   = [1, -1,  1, 1, 1, 1]
  offsets = [0,  90, 90, 0, 0, 0]   (degrees)

Usage:
    python scripts/serve_policy.py --ckpt_path run/checkpoints/step_70000.pt \\
        --host 0.0.0.0 --port 8765 --action_steps 16 eval_embodiment=so100

    python scripts/so100_policy_client.py --host localhost --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.policy_ws_client import PolicyWebSocketClient
from scripts.utils.mem_live_viz import (
    parse_bbox_cot,
    parse_affordance_cot,
    _BBOX_COLOR,
    _AFFORD_COLOR,
)

logger = logging.getLogger(__name__)

# ──────── CoT text helpers ────────

_LOC_RE            = re.compile(r"<loc\d+>")
# BBox section ends at "|" (template separator), "Action:", or end-of-string.
# Training format: "BBox: obj <loc0000>...; obj2 <loc>...|Action: ..."
# (no trailing period — the old r"BBox:[^.]*\." never matched)
_BBOX_SECTION_RE   = re.compile(r"BBox:.*?(?=\||Action:|$)", re.IGNORECASE | re.DOTALL)
_AFFORD_SECTION_RE = re.compile(
    r"(?:left_wrist|right_wrist|head)\s+camera:\s*\([^)]*\)", re.IGNORECASE
)

# Detect junk CoT: when pred_eov=false the model never stops generate_text,
# producing action tokens instead of meaningful CoT.  We filter these out by
# checking that the text contains recognizable structure (BBox, affordance,
# or enough alphabetic words to qualify as natural language).
_COT_VALID_MARKERS = re.compile(
    r"BBox\s*:|camera\s*:\s*\(|<loc\d+>", re.IGNORECASE
)


def _is_valid_cot(text: str) -> bool:
    """Return True if cot_text looks like genuine CoT rather than action token noise."""
    if not text or len(text.strip()) < 3:
        return False
    if _COT_VALID_MARKERS.search(text):
        return True
    # Count alphabetic words — action tokens are mostly digits/special chars
    alpha_words = [w for w in text.split() if w.isalpha() and len(w) > 1]
    return len(alpha_words) >= 3


def _clean_cot_for_display(cot_text: str) -> str:
    """Return only the plain CoT reasoning text for the text panel.

    Strips:
    - BBox section ("BBox: name <locXXXX>...|") — drawn on camera image instead
    - Affordance annotations ("left_wrist camera: (u, v)")
    - All <locXXXX> tokens
    - "predict bbox[...]" boilerplate prompt lines
    - "Action: ..." suffix
    - Template pipe separators "|"
    """
    if not cot_text:
        return ""
    text = _BBOX_SECTION_RE.sub("", cot_text)
    text = _AFFORD_SECTION_RE.sub("", text)
    text = _LOC_RE.sub("", text)
    # Strip "Action: ..." and everything after it (end of model output)
    text = re.sub(r"\bAction\s*:.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip "predict bbox[...]" boilerplate lines (prompt injected by BBoxCoTBuilder)
    text = re.sub(r"^predict\s+bbox[^\n]*\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    # Remove bare template pipe separators (keep surrounding text)
    text = text.replace("|", " ")
    # Clean up any leftover isolated punctuation from stripped sections
    text = re.sub(r"(?<!\w)\.\s*(?!\w)", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _cam_matches_view(cam_key: str, view: str) -> bool:
    """'right_wrist' CoT token matches 'wrist_right' cam key (word-set comparison)."""
    ck = set(cam_key.replace("-", "_").split("_"))
    v  = set(view.replace("-", "_").split("_"))
    return ck == v or ck.issuperset(v)


def _make_cam_placeholder(h: int = 360, w: int = 480) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for camera...", (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1, cv2.LINE_AA)
    return img


def _parse_chw_shape(text: str) -> tuple[int, int, int]:
    """Parse C,H,W shape string used for dummy camera padding."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected shape format C,H,W, e.g. 3,480,640")
    try:
        c, h, w = (int(x) for x in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("camera dummy shape must contain integers") from exc
    if c <= 0 or h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("camera dummy shape values must be positive")
    return c, h, w


def _parse_key_value_pairs(items: list[str]) -> dict[str, str]:
    """Parse CLI entries like SERVER_KEY:value into a dict."""
    result: dict[str, str] = {}
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"expected KEY:VALUE entry, got: {item!r}")
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"empty key in entry: {item!r}")
        result[key] = value
    return result


def _extract_missing_image_keys_from_error(error: object) -> list[str]:
    """Extract missing image keys from old serve_policy.py 400 error payloads.

    Old server examples:
      {"code": 400, "message": "images are missing shape_meta-defined key: {'wrist_left'}"}

    This lets the client learn server-required camera keys at runtime and
    zero-pad them on the next recompute instead of repeatedly failing.
    """
    if isinstance(error, dict):
        message = str(error.get("message", ""))
    else:
        message = str(error)
    if "images" not in message or "missing" not in message:
        return []
    return [k for k in re.findall(r"'([^']+)'", message) if k]


# ──────── joint constants ────────

_MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper",
]
JOINT_COUNT = len(_MOTOR_NAMES)

_SIGNS   = np.array([ 1, -1,  1,  1,  1,  1], dtype=np.float32)
_OFFSETS = np.array([ 0,  90, 90,  0,  0,  0], dtype=np.float32)

# Training data distribution center in model frame (from dataset_stats.json global_mean)
# Converted to arm frame: (model_mean - offsets) * signs
_HOME_ARM = np.array([
     3.1,   # shoulder_pan:  (3.1  - 0)  * 1  = 3.1
   -34.3,   # shoulder_lift: (124.3 - 90) * -1 = -34.3
    31.5,   # elbow_flex:    (121.5 - 90) * 1  = 31.5
    55.9,   # wrist_flex:    (55.9  - 0)  * 1  = 55.9
   -12.3,   # wrist_roll:    (-12.3 - 0)  * 1  = -12.3
    13.4,   # gripper:       (13.4  - 0)  * 1  = 13.4
], dtype=np.float32)

# FollowerArm inner rate limiter: max degrees moved per worker loop iteration
_MAX_INNER_DELTA = 10.0


# ──────── proprio OOD guard ────────────────────────────────────────────────

class ProprioGuard:
    """Clamp or soft-remap the model-frame state before sending to the server.

    Prevents out-of-distribution proprio inputs from collapsing the model's
    action predictions when the robot starts far from the training distribution.

    Two modes:
      clip   — hard-clamp each joint to [q01, q99] of training stats.
      zscore — pull joints that are > k·sigma from the training mean back to
               exactly k·sigma (values within k·sigma are untouched).

    All values are in *model frame* degrees (same coordinate system as
    arm_to_model() output and dataset_stats.json).
    """

    def __init__(self, mode: str, stats_path: str,
                 stats_embodiment: str = "so100",
                 stats_state_key: str = "right_arm",
                 zscore_k: float = 3.0):
        self.mode = mode
        self.zscore_k = float(zscore_k)
        self._q01: Optional[np.ndarray] = None
        self._q99: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        if mode == "none":
            return

        stats_file = Path(stats_path)
        if not stats_file.is_absolute():
            # Resolve relative to the repository parent used by the deployment layout.
            deploy_root = Path(__file__).resolve().parent.parent.parent
            stats_file = deploy_root / stats_path

        if not stats_file.exists():
            raise FileNotFoundError(
                f"ProprioGuard: stats file not found: {stats_file}\n"
                f"Set proprio_guard.stats_path in your client config."
            )

        raw = json.loads(stats_file.read_text())
        try:
            sk = raw[stats_embodiment]["state"][stats_state_key]
        except KeyError as e:
            raise KeyError(
                f"ProprioGuard: key {e} not found in {stats_file}. "
                f"Expected path: {stats_embodiment} → state → {stats_state_key}"
            ) from e

        self._q01  = np.array(sk["global_q01"],  dtype=np.float32)
        self._q99  = np.array(sk["global_q99"],  dtype=np.float32)
        self._mean = np.array(sk["global_mean"], dtype=np.float32)
        self._std  = np.array(sk["global_std"],  dtype=np.float32)

        logger.info(
            "[ProprioGuard] mode=%s  stats=%s\n"
            "  q01 : %s\n  q99 : %s\n  mean: %s\n  std : %s",
            mode, stats_file,
            np.round(self._q01,  1).tolist(),
            np.round(self._q99,  1).tolist(),
            np.round(self._mean, 1).tolist(),
            np.round(self._std,  1).tolist(),
        )

    def apply(self, state_model: np.ndarray) -> np.ndarray:
        """Return a (possibly clamped) copy of state_model. Input is model-frame degrees."""
        if self.mode == "none":
            return state_model

        state = state_model.copy()

        if self.mode == "clip":
            clipped = np.clip(state, self._q01, self._q99)
            n_clipped = int(np.sum(np.abs(clipped - state) > 1e-4))
            if n_clipped:
                logger.debug(
                    "[ProprioGuard/clip] clamped %d joint(s): raw=%s → clipped=%s",
                    n_clipped,
                    np.round(state, 1).tolist(),
                    np.round(clipped, 1).tolist(),
                )
            return clipped

        if self.mode == "zscore":
            # Compute signed deviation in units of std (safe: avoid div-by-zero)
            std_safe = np.where(self._std > 1e-6, self._std, 1.0)
            z = (state - self._mean) / std_safe
            # For dims where |z| > k, pull back to exactly ±k sigma
            z_clipped = np.clip(z, -self.zscore_k, self.zscore_k)
            remapped = self._mean + z_clipped * std_safe
            n_remapped = int(np.sum(np.abs(z) > self.zscore_k))
            if n_remapped:
                logger.debug(
                    "[ProprioGuard/zscore] remapped %d joint(s) (|z|>%.1f): "
                    "raw=%s → remapped=%s",
                    n_remapped, self.zscore_k,
                    np.round(state, 1).tolist(),
                    np.round(remapped, 1).tolist(),
                )
            return remapped

        raise ValueError(f"ProprioGuard: unknown mode {self.mode!r}. "
                         "Choose 'none', 'clip', or 'zscore'.")


def load_client_config(config_path: Optional[str]) -> dict:
    """Load YAML client config; return empty dict if no path given."""
    if not config_path:
        return {}
    try:
        import yaml  # PyYAML — available in the deploy venv
    except ImportError:
        raise ImportError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        )
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Client config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f) or {}


def build_proprio_guard(cfg: dict) -> ProprioGuard:
    """Instantiate ProprioGuard from the 'proprio_guard' section of the config."""
    pg_cfg = cfg.get("proprio_guard", {})
    return ProprioGuard(
        mode               = pg_cfg.get("mode", "none"),
        stats_path         = pg_cfg.get("stats_path", "run/dataset_stats.json"),
        stats_embodiment   = pg_cfg.get("stats_embodiment", "so100"),
        stats_state_key    = pg_cfg.get("stats_state_key", "right_arm"),
        zscore_k           = float(pg_cfg.get("zscore_k", 3.0)),
    )




def arm_to_model(s: np.ndarray) -> np.ndarray:
    """lerobot v3.0 degrees → training v2.1 degrees."""
    return _SIGNS * s + _OFFSETS


def model_to_arm(a: np.ndarray) -> np.ndarray:
    """training v2.1 degrees → lerobot v3.0 degrees."""
    return (a - _OFFSETS) * _SIGNS


def clip_action(target: np.ndarray, cur: np.ndarray, max_deg: float) -> np.ndarray:
    """Scale delta vector so no joint exceeds max_deg per tick."""
    delta   = target - cur
    biggest = float(np.max(np.abs(delta)))
    if biggest <= max_deg or biggest == 0.0:
        return target
    return cur + delta * (max_deg / biggest)


# ──────── FollowerArm ────────

class FollowerArm:
    """Single background thread owns the serial bus.

    Ported from molmoact2-so101/molmoact_so101/setup/robot.py.
    All external callers use set_target() / get_state() only.
    """

    def __init__(self, robot):
        self._robot       = robot
        self._target      = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._state       = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._target_lock = threading.Lock()
        self._state_lock  = threading.Lock()
        self._stop        = threading.Event()

        try:
            obs  = robot.get_observation()
            init = np.array([obs[f"{m}.pos"] for m in _MOTOR_NAMES], dtype=np.float32)
            self._target = init.copy()
            self._state  = init.copy()
            logger.info("[FollowerArm] init position: %s", np.round(init, 1).tolist())
        except Exception as e:
            logger.warning("[FollowerArm] could not read init position: %s", e)

        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="FollowerArmWorker"
        )
        self._thread.start()
        logger.info("[FollowerArm] worker started")

    def set_target(self, target: np.ndarray):
        with self._target_lock:
            self._target = target.astype(np.float32, copy=True)

    def get_state(self) -> np.ndarray:
        with self._state_lock:
            return self._state.copy()

    def move_to_home(self, home: np.ndarray = _HOME_ARM,
                     tol_deg: float = 2.0, timeout_s: float = 15.0,
                     max_deg_per_step: float = 5.0, step_interval: float = 0.02):
        """Interpolate toward home position with velocity limiting.

        Args:
            max_deg_per_step: max degrees any joint moves per interpolation step.
            step_interval: seconds between interpolation steps.
        """
        logger.info("[FollowerArm] homing to %s (arm-frame degrees)...", np.round(home, 1).tolist())
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            cur = self.get_state()
            err = float(np.max(np.abs(cur - home)))
            if err < tol_deg:
                logger.info("[FollowerArm] homing done (max_err=%.1f°)", err)
                return
            delta = np.clip(home - cur, -max_deg_per_step, max_deg_per_step)
            self.set_target(cur + delta)
            time.sleep(step_interval)
        logger.warning("[FollowerArm] homing timeout after %.0fs (max_err=%.1f°)",
                       timeout_s, float(np.max(np.abs(self.get_state() - home))))

    def _worker_loop(self):
        last_written = self._target.copy()
        while not self._stop.is_set():
            with self._target_lock:
                target = self._target.copy()

            delta        = np.clip(target - last_written, -_MAX_INNER_DELTA, _MAX_INNER_DELTA)
            last_written = last_written + delta

            try:
                self._robot.send_action(
                    {f"{m}.pos": float(last_written[i]) for i, m in enumerate(_MOTOR_NAMES)}
                )
            except Exception as e:
                logger.warning("[FollowerArm] send_action: %s", e)

            try:
                obs   = self._robot.get_observation()
                state = np.array([obs[f"{m}.pos"] for m in _MOTOR_NAMES], dtype=np.float32)
                with self._state_lock:
                    self._state = state
            except Exception as e:
                logger.warning("[FollowerArm] get_observation: %s", e)

    def disconnect(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._robot.disconnect()
        except Exception:
            pass


# ──────── CameraWorker ────────

class CameraWorker:
    """Background-thread OpenCV camera capture."""

    def __init__(self, index: int, width: int = 640, height: int = 480, fps: int = 30):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._lock   = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop   = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"Camera-{index}"
        )
        self._thread.start()
        logger.info("[Camera-%d] started (%dx%d @ %dfps)", index, width, height, fps)

    def _loop(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame  # BGR HWC uint8

    def read_bgr(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read_rgb_chw(self) -> np.ndarray | None:
        """RGB CHW uint8 — format expected by the server."""
        frame = self.read_bgr()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


# ──────── lerobot helpers ────────

def _load_robot_class():
    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
        return SOFollower
    except ImportError:
        pass
    try:
        from lerobot.robots.manipulator import ManipulatorRobot
        return ManipulatorRobot
    except ImportError:
        pass
    raise ImportError("lerobot not found.")


def _load_so100_config_class():
    try:
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        return SOFollowerRobotConfig
    except (ImportError, AttributeError):
        pass
    for mod, cls in [
        ("lerobot.configs.robots.so100", "So100RobotConfig"),
        ("lerobot.configs.robots",       "So100RobotConfig"),
        ("lerobot.common.robot_devices.robots.configs", "So100RobotConfig"),
    ]:
        try:
            import importlib
            return getattr(importlib.import_module(mod), cls)
        except (ImportError, AttributeError):
            pass
    raise ImportError("Could not import So100RobotConfig.")


def _connect_follower(args) -> FollowerArm:
    """Connect lerobot robot (no cameras) and wrap in FollowerArm."""
    RobotClass  = _load_robot_class()
    RobotConfig = _load_so100_config_class()

    import inspect
    params = inspect.signature(RobotConfig.__init__).parameters
    if "port" in params:
        cfg = RobotConfig(
            port=args.robot_port,
            id=args.robot_id,
            cameras={},          # cameras managed separately by CameraWorker
            use_degrees=True,
        )
    else:
        cfg = RobotConfig()

    robot = RobotClass(cfg)
    robot.connect()
    logger.info("[Robot] connected on %s", args.robot_port)
    return FollowerArm(robot)


# ──────── InferenceProducer ────────

class InferenceProducer(threading.Thread):
    """Async loop at action_fps Hz: send obs/cache request → set_target().

    Follows the server's need_obs flag:
    - need_obs=True  → send full obs (images + state) → server infers
    - need_obs=False → send {}                        → server returns cached step
    No ring buffer, no consumer thread, no drain loop.
    """

    def __init__(self, *, follower: FollowerArm,
                 camera_workers: dict[str, CameraWorker],
                 ws_uri: str, task: str,
                 action_fps: float, max_step_deg: float,
                 proprio_guard: Optional["ProprioGuard"] = None,
                 expected_camera_keys: Optional[list[str]] = None,
                 dummy_camera_shape: tuple[int, int, int] = (3, 480, 640)):
        super().__init__(daemon=True, name="InferenceProducer")
        self.follower       = follower
        self.camera_workers = camera_workers
        self.ws_uri         = ws_uri
        self.action_fps     = action_fps
        self.max_step_deg   = max_step_deg
        self.proprio_guard  = proprio_guard or ProprioGuard(mode="none", stats_path="")
        self.expected_camera_keys = list(expected_camera_keys or camera_workers.keys())
        self.dummy_camera_shape   = tuple(int(x) for x in dummy_camera_shape)
        self._missing_cam_warned: set[str] = set()
        self._stop_event    = threading.Event()
        self._frames_lock   = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}   # BGR HWC for display
        self._need_full_obs = True
        self._step_count    = 0
        # CoT — written by asyncio thread, read by main thread
        self._cot_lock = threading.Lock()
        self._cot_text: str | None = None
        # Task — written by stdin thread, read by asyncio thread
        self._task_lock    = threading.Lock()
        self._task         = task
        self._task_changed = False

    def stop(self):
        self._stop_event.set()

    def _add_expected_camera_keys(self, keys: list[str]) -> bool:
        """Add server-required camera keys so later obs sends zero padding for them."""
        added = False
        for key in keys:
            if key and key not in self.expected_camera_keys:
                self.expected_camera_keys.append(key)
                added = True
        if added:
            logger.warning(
                "[Producer] learned missing server camera key(s) %s → will zero-pad from next request",
                keys,
            )
        return added

    def latest_frames(self) -> dict[str, np.ndarray]:
        with self._frames_lock:
            return dict(self._frames)

    def latest_cot(self) -> str | None:
        with self._cot_lock:
            return self._cot_text

    def set_task(self, task: str) -> None:
        """Called from stdin thread; forces server recompute on next cycle."""
        task = task.strip()
        if not task:
            return
        with self._task_lock:
            self._task = task
            self._task_changed = True
        logger.info("[Producer] set_task: %r", task)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_run())
        finally:
            loop.close()

    async def _async_run(self):
        async with PolicyWebSocketClient(self.ws_uri) as client:
            logger.info("[Producer] connected to %s", self.ws_uri)
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    await self._one_cycle(client)
                except Exception as e:
                    logger.error("[Producer] %s: %s", type(e).__name__, e, exc_info=True)
                    self._need_full_obs = True
                    await asyncio.sleep(0.5)
                    continue
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, 1.0 / self.action_fps - elapsed))

    async def _one_cycle(self, client: PolicyWebSocketClient):
        # Pick up task update from stdin thread (executed in asyncio thread)
        with self._task_lock:
            current_task = self._task
            if self._task_changed:
                self._task_changed = False
                self._need_full_obs = True
                logger.info("[Producer] task changed → RECOMPUTE: %r", current_task)

        state_arm = self.follower.get_state()

        if self._need_full_obs:
            state_model = arm_to_model(state_arm)
            # Apply proprio OOD guard (clip or zscore) before sending to server.
            # guard.apply() is a no-op when mode="none".
            state_model_guarded = self.proprio_guard.apply(state_model)
            logger.info(
                "[Producer] RECOMPUTE  arm_state=%s  model_state=%s%s",
                np.round(state_arm, 1).tolist(),
                np.round(state_model, 1).tolist(),
                "" if np.allclose(state_model, state_model_guarded, atol=0.01)
                else f"  guarded={np.round(state_model_guarded, 1).tolist()}",
            )
            # Build full obs with images.  Keep compatibility with the old
            # strict server: every expected image key must be present.  If a
            # camera is missing/not ready, send a uint8 black CHW image instead
            # of omitting the key and letting serve_policy.py raise ValueError.
            images: dict[str, np.ndarray] = {}
            frames: dict[str, np.ndarray] = {}
            for srv_key in self.expected_camera_keys:
                cam = self.camera_workers.get(srv_key)
                chw = cam.read_rgb_chw() if cam is not None else None
                if chw is None:
                    images[srv_key] = np.zeros(self.dummy_camera_shape, dtype=np.uint8)
                    if srv_key not in self._missing_cam_warned:
                        logger.warning(
                            "[Producer] camera '%s' missing/not ready → using zero padding shape=%s",
                            srv_key, self.dummy_camera_shape,
                        )
                        self._missing_cam_warned.add(srv_key)
                    continue

                images[srv_key] = chw
                bgr = cam.read_bgr()
                if bgr is not None:
                    frames[srv_key] = bgr

            # Preserve any extra camera-index keys not listed in camera-map.
            for srv_key, cam in self.camera_workers.items():
                if srv_key in images:
                    continue
                chw = cam.read_rgb_chw()
                if chw is None:
                    continue
                images[srv_key] = chw
                bgr = cam.read_bgr()
                if bgr is not None:
                    frames[srv_key] = bgr

            with self._frames_lock:
                self._frames = frames

            raw_obs = {
                "images":          images,
                "state":           {"right_arm": state_model_guarded},
                "task":            current_task,
                "embodiment_type": "so100",
                "frequency":       float(self.action_fps),
            }
        else:
            raw_obs = {}   # server serves from cached chunk

        response = await client.infer(raw_obs)

        # Capture CoT text from server response (filter out action-token noise)
        cot = response.get("cot_text")
        if cot is not None:
            if _is_valid_cot(cot):
                with self._cot_lock:
                    self._cot_text = cot
            else:
                with self._cot_lock:
                    self._cot_text = None

        if "error" in response:
            missing_image_keys = _extract_missing_image_keys_from_error(response["error"])
            if missing_image_keys:
                self._add_expected_camera_keys(missing_image_keys)
            logger.error("[Producer] server error: %s", response["error"])
            self._need_full_obs = True
            return

        self._need_full_obs = bool(response.get("need_obs", True))

        action_model = np.asarray(response["action"]["right_arm"], dtype=np.float32)
        action_arm   = model_to_arm(action_model)
        target       = clip_action(action_arm, state_arm, self.max_step_deg)

        self.follower.set_target(target)
        self._step_count += 1

        if self._step_count % 15 == 1:
            logger.info(
                "[Producer] step=%d  a=%s  Δ=%s  need_obs=%s",
                self._step_count,
                np.round(target, 1).tolist(),
                np.round(target - state_arm, 2).tolist(),
                self._need_full_obs,
            )


# ──────── dashboard rendering ────────

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_TASK_COLOR = (0, 255, 255)     # yellow
_COT_COLOR  = (180, 255, 180)   # light green
_STEP_COLOR = (255, 200, 100)   # orange
_HINT_COLOR = (90,  90,  90)    # dim gray
_PANEL_BG   = (20,  20,  20)    # near-black
_CAM_H      = 360
_TEXT_H     = 180
_INPUT_H          = 48
_INPUT_BG         = (28,  28,  28)
_INPUT_ON_BORDER  = (70, 170, 70)
_INPUT_OFF_BORDER = (50,  50,  50)
_INPUT_LABEL_CLR  = (120, 190, 120)
_INPUT_TEXT_CLR   = (215, 215, 215)


def _process_key(raw_key: int, buf: str) -> tuple[str, str | None]:
    """Convert cv2.waitKey() value → (new_buf, submitted_text | None).

    Returns submitted_text (non-empty stripped string) when Enter is pressed,
    None otherwise.  Escape clears the buffer without submitting.
    """
    if raw_key == -1:
        return buf, None
    key = raw_key & 0xFF
    if key in (13, 10):        # Enter
        s = buf.strip()
        return "", s or None
    if key in (8, 127):        # Backspace / Delete
        return buf[:-1], None
    if key == 27:              # Escape — clear only
        return "", None
    if 32 <= key <= 126:       # printable ASCII
        return buf + chr(key), None
    return buf, None


def _render_input_box(width: int, label: str, buf: str, active: bool) -> np.ndarray:
    """Render a single-line text-input panel."""
    panel = np.full((_INPUT_H, width, 3), _INPUT_BG, dtype=np.uint8)
    border = _INPUT_ON_BORDER if active else _INPUT_OFF_BORDER
    cv2.rectangle(panel, (3, 3), (width - 4, _INPUT_H - 4), border, 1, cv2.LINE_AA)
    if not active:
        cv2.putText(panel, "Waiting…", (12, _INPUT_H // 2 + 6),
                    _FONT, 0.48, _INPUT_OFF_BORDER, 1, cv2.LINE_AA)
        return panel
    (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.52, 1)
    ty = _INPUT_H // 2 + lh // 2
    cv2.putText(panel, label, (10, ty), _FONT, 0.52, _INPUT_LABEL_CLR, 1, cv2.LINE_AA)
    x0    = 10 + lw + 6
    avail = width - x0 - 14
    display = buf + "|"
    (tw, _), _ = cv2.getTextSize(display, _FONT, 0.52, 1)
    while tw > avail and len(display) > 2:   # scroll left if text overflows
        display = display[1:]
        (tw, _), _ = cv2.getTextSize(display, _FONT, 0.52, 1)
    cv2.putText(panel, display, (x0, ty), _FONT, 0.52, _INPUT_TEXT_CLR, 1, cv2.LINE_AA)
    return panel


def _wrap_text(text: str, max_w: int, font, scale: float, thickness: int) -> list[str]:
    """Word-wrap text to fit within max_w pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip() if current else word
        (tw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if tw > max_w and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_composite_frame(
    frames: dict[str, np.ndarray],
    cam_keys: list[str],
    task: str,
    cot_text: str | None,
    step_count: int,
    need_obs: bool,
    phase: str = "infer",          # "homing" | "waiting_task" | "infer"
    input_buffer: str = "",
) -> np.ndarray:
    """Assemble unified dashboard: cameras (top) + text panel (bottom)."""
    _SCALE_TASK = 0.60
    _SCALE_COT  = 0.55
    _MARGIN     = 10

    # ── camera row ──────────────────────────────────────────────────
    cam_panels = []
    for cam_key in cam_keys[:2]:
        bgr = frames.get(cam_key)
        if bgr is None:
            panel = _make_cam_placeholder(_CAM_H)
        else:
            h0, w0 = bgr.shape[:2]
            nw = max(1, int(round(w0 * _CAM_H / h0)))
            panel = cv2.resize(bgr, (nw, _CAM_H)) if h0 != _CAM_H else bgr.copy()

        # bbox + affordance overlay (only during inference)
        if cot_text and phase == "infer":
            h, w = panel.shape[:2]
            if "exterior" in cam_key:
                for name, x1, y1, x2, y2 in parse_bbox_cot(cot_text):
                    p1 = (int(round(x1 * w)), int(round(y1 * h)))
                    p2 = (int(round(x2 * w)), int(round(y2 * h)))
                    cv2.rectangle(panel, p1, p2, _BBOX_COLOR, 2, cv2.LINE_AA)
                    cv2.putText(panel, name, (p1[0] + 3, max(14, p1[1] - 4)),
                                _FONT, 0.5, _BBOX_COLOR, 1, cv2.LINE_AA)
            for view, u01, v01 in parse_affordance_cot(cot_text):
                if not _cam_matches_view(cam_key, view):
                    continue
                cx, cy = int(round(u01 * w)), int(round(v01 * h))
                cv2.drawMarker(panel, (cx, cy), _AFFORD_COLOR,
                               cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
                cv2.circle(panel, (cx, cy), 10, _AFFORD_COLOR, 1, cv2.LINE_AA)

        cv2.putText(panel, cam_key, (6, 22), _FONT, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cam_panels.append(panel)

    if not cam_panels:
        cam_panels.append(_make_cam_placeholder(_CAM_H))

    # pad panels to uniform height then hstack
    max_h = max(p.shape[0] for p in cam_panels)
    uniform = []
    for p in cam_panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        uniform.append(p)
    top_row = np.hstack(uniform)
    total_w = top_row.shape[1]

    # ── text panel ───────────────────────────────────────────────────
    text_panel = np.full((_TEXT_H, total_w, 3), _PANEL_BG, dtype=np.uint8)
    (_, lh), baseline = cv2.getTextSize("Ag", _FONT, _SCALE_COT, 1)
    line_stride = lh + baseline + 5
    avail_w     = total_w - 2 * _MARGIN
    y           = _MARGIN + lh

    def put_line(text, color, scale=_SCALE_COT):
        nonlocal y
        if y + baseline > _TEXT_H - line_stride * 2 - 4:
            return
        cv2.putText(text_panel, text, (_MARGIN, y), _FONT, scale, color, 1, cv2.LINE_AA)
        y += line_stride

    if phase == "homing":
        put_line("Homing to start position...", (100, 200, 255), scale=_SCALE_TASK)
        put_line("Please wait for the arm to reach the home pose.", _HINT_COLOR)
    elif phase == "waiting_task":
        put_line("Homing complete.", (100, 255, 100), scale=_SCALE_TASK)
        put_line("Type a task in the input box below and press Enter.", _HINT_COLOR)
    else:
        put_line(f"Task: {task}" if task else "Task: (none)", _TASK_COLOR, scale=_SCALE_TASK)

        cleaned = _clean_cot_for_display(cot_text or "")
        if cleaned:
            for line in _wrap_text(cleaned, avail_w, _FONT, _SCALE_COT, 1):
                put_line(line, _COT_COLOR)

        # step status — anchored to bottom
        status = "INFER" if need_obs else "CACHE"
        step_y = _TEXT_H - line_stride - baseline - 4
        cv2.putText(text_panel, f"Step: {step_count}  [{status}]",
                    (_MARGIN, step_y), _FONT, _SCALE_COT, _STEP_COLOR, 1, cv2.LINE_AA)

    # ── input box ────────────────────────────────────────────────────
    input_active = phase in ("waiting_task", "infer")
    label = "Enter task> " if phase == "waiting_task" else "Update task> "
    input_panel = _render_input_box(total_w, label, input_buffer, active=input_active)

    return np.vstack([top_row, text_panel, input_panel])


# ──────── stdin input thread ────────

def _stdin_input_thread(producer: "InferenceProducer", stop: threading.Event) -> None:
    """Daemon thread: read new task lines from stdin for multi-round control."""
    print("[Dashboard] Running. Type a new task and press Enter to update.")
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except (OSError, ValueError):
            break
        if not line:   # EOF (Ctrl-D)
            break
        task = line.strip()
        if task:
            producer.set_task(task)
            print(f"[Dashboard] Task updated: {task!r}")


# ──────── task input thread ────────

# ──────── main ────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(description="SO100 policy client — G05 server")
    p.add_argument("--host",         default="localhost")
    p.add_argument("--port",         type=int,   default=8765)
    p.add_argument("--robot-port",   default="/dev/ttyACM0")
    p.add_argument("--robot-id",     default="g05_so100_follower")
    p.add_argument("--action-fps",   type=float, default=15.0,
                   help="Control frequency Hz — must match server training FPS (default 15)")
    p.add_argument("--max-step-deg", type=float, default=10.0,
                   help="Per-tick safety cap in degrees (default 10)")
    p.add_argument("--camera-map",
                   nargs="+", default=["exterior:exterior", "wrist_right:wrist_right"],
                   metavar="SERVER_KEY:LEROBOT_CAM",
                   help="Server key → lerobot camera name (for reference only)")
    p.add_argument("--camera-index",
                   nargs="+", default=["exterior:2", "wrist_right:0"],
                   metavar="SERVER_KEY:CV2_INDEX",
                   help="Server key → OpenCV device index. e.g. exterior:2 wrist_right:0")
    p.add_argument("--dummy-camera-shape", type=_parse_chw_shape, default=(3, 480, 640),
                   metavar="C,H,W",
                   help="CHW uint8 zero-padding shape for missing/not-ready cameras (default: 3,480,640)")
    p.add_argument("--dummy-camera-key", nargs="*", default=["wrist_left"],
                   metavar="SERVER_KEY",
                   help=(
                       "Additional server image keys to always zero-pad when no physical camera is configured. "
                       "Default includes wrist_left for old SO100 checkpoints whose shape_meta expects it."
                   ))
    p.add_argument("--no-display", action="store_true",
                   help="Disable camera windows")
    p.add_argument("--dry-run", action="store_true",
                   help="Infer but do NOT call set_target")
    p.add_argument("--no-home", action="store_true",
                   help="Skip homing to training-distribution center at startup")
    p.add_argument("--config", default=None, metavar="PATH",
                   help=(
                       "Path to client YAML config file. "
                       "Supports proprio_guard section (mode: none|clip|zscore). "
                       "See client_config.yaml for a fully annotated example."
                   ))
    args = p.parse_args()

    camera_map   = _parse_key_value_pairs(args.camera_map)
    camera_index = _parse_key_value_pairs(args.camera_index)
    # The server only validates image keys, not lerobot camera names.  Use
    # Use camera-map + camera-index + explicit dummy-camera-key as the expected
    # server-key list. Keys without a physical CameraWorker are zero-padded.
    # If old serve_policy.py reports more missing keys, the producer learns
    # them at runtime and zero-pads them on the next recompute.
    expected_camera_keys = list(dict.fromkeys([*camera_map.keys(), *camera_index.keys(), *args.dummy_camera_key]))
    ws_uri       = f"ws://{args.host}:{args.port}"

    # ── Load client config + build proprio guard ────────────────────
    client_cfg    = load_client_config(args.config)
    proprio_guard = build_proprio_guard(client_cfg)

    # ── Phase 1: connect robot ──────────────────────────────────────
    follower = _connect_follower(args)

    # ── Phase 2: start cameras immediately ─────────────────────────
    camera_workers: dict[str, CameraWorker] = {}
    for srv_key, idx_str in camera_index.items():
        try:
            idx = int(idx_str)
        except ValueError:
            idx = idx_str
        camera_workers[srv_key] = CameraWorker(idx)

    # Wait for cameras to produce first frame
    time.sleep(0.5)

    cam_keys = expected_camera_keys or list(camera_workers.keys())
    show = not args.no_display

    # ── Phase 3: home arm (blocking in background thread, show live camera) ──
    homing_done = threading.Event()

    def _do_home():
        if not args.no_home:
            follower.move_to_home()
        homing_done.set()

    home_thread = threading.Thread(target=_do_home, daemon=True, name="HomingThread")
    home_thread.start()

    # Show camera + "homing" overlay until home completes
    if show:
        while not homing_done.is_set():
            frames = {k: cw.read_bgr() for k, cw in camera_workers.items()}
            composite = build_composite_frame(
                frames=frames, cam_keys=cam_keys,
                task="", cot_text=None,
                step_count=0, need_obs=True,
                phase="homing", input_buffer="",
            )
            cv2.imshow("G05 Dashboard", composite)
            raw_key = cv2.waitKey(33)
            if raw_key != -1 and (raw_key & 0xFF) == 27:  # Escape → quit
                home_thread.join(timeout=1.0)
                for cw in camera_workers.values():
                    cw.release()
                follower.disconnect()
                cv2.destroyAllWindows()
                return
    else:
        homing_done.wait()

    home_thread.join(timeout=1.0)
    logger.info("[Main] homing complete.")

    # ── Phase 4: wait for task input ───────────────────────────────
    # Display mode: type in the cv2 window input box, Enter to submit.
    # No-display mode: blocking stdin.
    task = ""
    input_buffer = ""

    if show:
        while True:
            frames = {k: cw.read_bgr() for k, cw in camera_workers.items()}
            composite = build_composite_frame(
                frames=frames, cam_keys=cam_keys,
                task="", cot_text=None,
                step_count=0, need_obs=True,
                phase="waiting_task", input_buffer=input_buffer,
            )
            cv2.imshow("G05 Dashboard", composite)
            raw_key = cv2.waitKey(33)
            input_buffer, submitted = _process_key(raw_key, input_buffer)
            if submitted:
                task = submitted
                break
            if raw_key != -1 and (raw_key & 0xFF) == 27:  # Escape → quit
                for cw in camera_workers.values():
                    cw.release()
                follower.disconnect()
                cv2.destroyAllWindows()
                return
    else:
        print("[Dashboard] Homing complete. Type a task description and press Enter.")
        task = input("Task> ").strip()
    if not task:
        print("No task given, exiting.")
        for cw in camera_workers.values():
            cw.release()
        follower.disconnect()
        if show:
            cv2.destroyAllWindows()
        return

    # ── Phase 5: start inference ────────────────────────────────────
    if args.dry_run:
        class _DryRunFollower:
            def get_state(self):       return follower.get_state()
            def set_target(self, t):   logger.info("[DryRun] would set_target %s", np.round(t, 1))
            def disconnect(self):      follower.disconnect()
        active_follower = _DryRunFollower()
    else:
        active_follower = follower

    producer = InferenceProducer(
        follower=active_follower,
        camera_workers=camera_workers,
        ws_uri=ws_uri,
        task=task,
        action_fps=args.action_fps,
        max_step_deg=args.max_step_deg,
        proprio_guard=proprio_guard,
        expected_camera_keys=expected_camera_keys,
        dummy_camera_shape=args.dummy_camera_shape,
    )

    input_buffer = ""   # cv2 keyboard input buffer for task updates

    try:
        producer.start()

        # No-display mode: keep stdin thread for task updates
        stop_stdin = threading.Event()
        if not show:
            threading.Thread(
                target=_stdin_input_thread, args=(producer, stop_stdin),
                daemon=True, name="StdinInputThread"
            ).start()

        while True:
            if show:
                frames   = producer.latest_frames()
                if not frames:
                    frames = {k: cw.read_bgr() for k, cw in camera_workers.items()}
                cot_text = producer.latest_cot()
                step     = producer._step_count
                need_obs = producer._need_full_obs
                with producer._task_lock:
                    current_task = producer._task
                composite = build_composite_frame(
                    frames=frames, cam_keys=cam_keys,
                    task=current_task, cot_text=cot_text,
                    step_count=step, need_obs=need_obs,
                    phase="infer", input_buffer=input_buffer,
                )
                cv2.imshow("G05 Dashboard", composite)
                raw_key = cv2.waitKey(1)
                input_buffer, submitted = _process_key(raw_key, input_buffer)
                if submitted:
                    producer.set_task(submitted)
                elif raw_key != -1 and (raw_key & 0xFF) == 27:  # Escape → quit
                    break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_stdin.set()
        producer.stop()
        producer.join(timeout=3.0)
        for cw in camera_workers.values():
            cw.release()
        follower.disconnect()
        if show:
            cv2.destroyAllWindows()
        logger.info("Done.")


if __name__ == "__main__":
    main()
