# r1lite — R1_LITE Deployment Client

This directory contains the R1_LITE real-robot deployment client.
It is a pure client node: it collects raw observations via ROS2, sends them to the inference server over WebSocket, receives raw actions, and executes trajectories.

**This project does not include model loading, inference, or pre/post-processing logic.** All of that is handled by the server (`scripts/serve_policy.py`). The client has **zero torch dependency** — all data is numpy ndarray.

> Similar to `experiments/so100` and `experiments/droid` — all are deployment components that connect to `serve_policy.py`, targeting different robots.

## Architecture

```
  Robot (R1_LITE)                            GPU Server
  ───────────────                            ──────────
  experiments/r1lite/run.py   ──ws──►   scripts/serve_policy.py
  (ROS2 collect → send obs)   ◄──ws──   (G05 model inference + pre/post-processing)
  (receive action → smooth → publish)
```

## Environment Setup

**r1lite runs inside Python 3.10 + ROS2.**

The client dependencies are declared in this directory's `pyproject.toml` (`numpy / opencv-python / msgpack / websockets / typing-extensions / toml / loguru`). The repository root G05 environment also includes `websockets` after `uv sync`. The only non-pip requirement is `rclpy`, which is injected via `source /opt/ros/humble/setup.bash`.

### 1. Install G05 Environment

Follow the main repository README to set up the G05 environment with `uv sync`. This creates a `.venv` at the repository root with Python 3.10 and all required packages.

> **Important**: The G05 venv uses Python 3.10, which is required by ROS2 Humble. Do **not** create a separate venv in this directory — `uv sync` here would pull the latest Python (3.12), which is incompatible with `rclpy`.

### 2. ROS2

ROS2 Humble (or compatible) is required, with the following packages:

```
rclpy, sensor_msgs, geometry_msgs, builtin_interfaces, std_msgs
```

These are typically included with `ros-humble-desktop`. See [ROS2 Installation](https://docs.ros.org/en/humble/Installation.html).

`rclpy` is **not a pip package** — it is injected via `source /opt/ros/humble/setup.bash`.

---

## Client-Server Protocol

```
 Client                                 Server (serve_policy.py)
 ──────                                 ────────────────────────
 ROS2 gather_obs() → raw obs
   images: {cam: ndarray CHW uint8}
   state:  {part: ndarray 1D float32}
   task:   str
   embodiment_type: str (for mixture)
          ──── msgpack over WebSocket ──►  receive raw obs
                                           build_obs_dict(obs, processor)
                                             ├─ pad dummy action zeros
                                             ├─ pad is_pad masks
                                             └─ numpy → torch → cuda
                                           processor.preprocess(obs_dict)
                                           unsqueeze(0) → batch dim
                                           policy.predict_action(batch)
                                           → cpu
                                           processor.postprocess(batch)
          ◄──── msgpack ────               send action dict
 receive action dict
   {body_part: ndarray [T, D] float32}
 actions_dict_to_trajectory()
 publish via ROS2
```

### Client Sends (raw obs)

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `images` | `dict[str, ndarray]` | `{cam_name: (C, H, W) uint8}` | RGB images per camera (CHW format) |
| `state` | `dict[str, ndarray]` | `{part_name: (D,) float32}` | Joint states per body part |
| `task` | `str` | — | Natural language instruction (injected by InstructionManager) |
| `embodiment_type` | `str` (optional) | — | Specified for multi-embodiment mixture |

### Client Receives (raw action)

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `{body_part}` | `ndarray` | `(T, D) float32` | T-step actions per body part, raw physical units (radians/meters) |

The server may also return `server_timing` and other metadata, which the client ignores.

### Key Principles

1. **Client does not normalize / denormalize** — all normalization, coordinate transforms, and merging/splitting happen on the server
2. **Client does not load processor / model config** — no dependency on server-side G05 packages, `omegaconf`, or `torch`
3. **Client does not add batch dimension** — `gather_obs()` returns a single frame; server calls `unsqueeze(0)`
4. **Serialization protocol** — msgpack + custom numpy encoding (see `utils/websocket/msgpack.py`)

---

## Data Flow

```
instruction.txt ──► InstructionManager
                          │
                          ▼
ROS2 Topics ──► Ros2Bridge.gather_obs() ──► obs dict (numpy)
                                                │
                          InstructionManager.get_instruction(obs)
                          inject obs["task"]    │
                                                ▼
                          WebSocketClientEngine.predict_action(obs)
                            ├─ msgpack pack
                            ├─ ws.send()
                            ├─ ws.recv()
                            └─ msgpack unpack ──► action dict (numpy)
                                                      │
                          actions_dict_to_trajectory() │
                                                      ▼
                          Ros2Bridge.publish_action() ──► ROS2 Topics
```

---

## Directory Structure

```
r1lite/
├── run.py                              # Entry point: load config.toml → Scheduler.run()
├── start.sh                            # Environment setup (ROS2 + venv)
├── config.toml                         # Default configuration
├── pyproject.toml                      # Dependency declaration
│
├── scheduler/
│   ├── scheduler.py                    # Main loop: gather → instruct → predict → step
│   └── instruction/
│       ├── instruction.py              # InstructionManager (file/VLM instruction source)
│       └── instruction.txt             # Current instruction file
│
├── core/
│   ├── communication/
│   │   ├── ros2_bridge.py              # ROS2 bridge: subscribe sensors, publish actions, gather_obs()
│   │   ├── robot_topics.py             # ROS2 topic names and QoS configuration
│   │   └── message_queue.py            # Thread-safe message buffer
│   └── inference/
│       ├── inference_engine.py         # Abstract base class (connect / predict_action)
│       ├── websocket_engine.py         # WebSocket client implementation
│       └── factory.py                  # create_inference_engine()
│
├── utils/
│   ├── websocket/
│   │   └── msgpack.py                  # numpy ↔ msgpack serialization
│   ├── message/
│   │   ├── datatype.py                 # RobotAction / Trajectory data classes
│   │   ├── message_convert.py          # ROS2 msg ↔ numpy conversion, actions_dict_to_trajectory
│   │   └── bbox_utils.py              # bbox crop/format utilities (VLM mode)
│   ├── viz.py                          # CoT bbox visualization (--visualize)
│   └── episode_recorder.py             # Episode recording (--record)
│
├── scripts/
│   ├── launch_client_8080.sh           # Launch client on port 8080
│   ├── launch_client_8082.sh           # Launch client on port 8082
│   ├── run.sh                          # Simplified launch script
│   └── record.sh                       # Record ROS2 bag
│
└── tests/                              # Unit tests (runs locally without ROS2 or server)
    ├── conftest.py                     # PYTHONPATH + ROS2/websockets stubs
    ├── test_msgpack.py
    ├── test_factory.py
    ├── test_datatype.py
    └── test_instruction_manager.py
```

---

## Quick Start

### 1. Configuration

Edit `config.toml` to point `[websocket]` to the inference server:

```toml
[robot]
hardware = "R1_LITE"

[basic]
control_frequency = 15.0    # Control frequency (Hz)
step_mode = "sync"          # "sync" | "async"
action_steps = 16           # Action steps per inference call

[websocket]
host = "10.0.0.100"         # ← inference server IP
port = 8080                 # ← inference server port

[instruction]
use_vlm = false
```

### 2. Start the Inference Server (GPU side)

```bash
cd /path/to/g05-repo
PYTHONPATH="src:${PYTHONPATH:-}" python scripts/serve_policy.py \
  --ckpt_path /path/to/model/checkpoints/model_state_dict.pt \
  --host 0.0.0.0 \
  --port 8080 \
  --device cuda \
  eval_embodiment=galaxea_r1lite \
  model.model_weights_to_bf16=true \
  model.use_torch_compile=true \
  model.model_arch.attn_implementation=sdpa \
  model.model_arch.discrete_action=true \
  model.model_arch.continuous_action=false
```

> **Note**: `--ckpt_path` must point to a specific checkpoint file (e.g., `checkpoints/model_state_dict.pt`), not the model root directory.

### 3. Start the Client (robot side)

```bash
# Activate environment
source /opt/ros/humble/setup.bash             # inject rclpy
source /path/to/g05-repo/.venv/bin/activate  # G05 venv

# Launch
cd experiments/r1lite
python run.py --config config.toml
```

Or use the launch script:

```bash
bash experiments/r1lite/scripts/launch_client_8080.sh
```

### 4. Run Tests (no ROS2 or server required)

```bash
source /path/to/g05-repo/.venv/bin/activate
cd experiments/r1lite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v
```

Tests automatically stub out ROS2 and websockets dependencies via `conftest.py`. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prevents unrelated ROS2 pytest plugins from being auto-loaded from the shell environment.

---

## Dependencies

### Runtime (7 packages, all included in G05 venv)

| Package | Purpose |
|---------|---------|
| `numpy` | obs/action data format |
| `opencv-python` | Image decoding (ROS2 CompressedImage → RGB numpy) |
| `msgpack` | obs/action serialization, WebSocket transport protocol |
| `websockets` | WebSocket client, connects to inference server |
| `typing-extensions` | `@override` decorator |
| `toml` | Read config.toml |
| `loguru` | Logging |

### Optional

| Extra | Package | When needed |
|-------|---------|-------------|
| `dev` | `pytest` | Running tests |

VLM bbox data is consumed from the ROS2 `hs/vlm_out2vla` topic; the client no longer includes a remote bbox dependency.

### Not Required

The following are server-side dependencies — the client does **not** need them:

`torch`, server-side G05 model packages, `omegaconf`, `accelerate`, `hydra`, `lightning`, `diffusers`, `timm`, `tensorboard`, `h5py`, `zarr`, etc.
