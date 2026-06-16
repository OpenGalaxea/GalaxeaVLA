# r1lite — R1_LITE 真机部署客户端

本目录提供 **R1_LITE 真机部署客户端**。
它是一个纯 client 节点：通过 ROS2 采集 raw obs，经 WebSocket 发送到远程推理服务器，接收 raw action 并执行轨迹。

**本项目不包含模型加载、推理、预处理/后处理逻辑。** 这些全部由 server 端（`scripts/serve_policy.py`）负责。Client 端**零 torch 依赖**，所有数据均为 numpy ndarray。

> 与 `experiments/so100`、`experiments/droid` 同类——都是连 `serve_policy.py` 的部署端组件，只是面向不同机器人。

## 在 G05 部署链路中的定位

```
  机器人端 (R1_LITE)                         GPU 端
  ─────────────────                         ──────
  experiments/r1lite/run.py   ──ws──►   scripts/serve_policy.py
  (ROS2 采集 → 发 obs)        ◄──ws──   (g05 模型推理 + 前/后处理)
  (收 action → 轨迹平滑 → 发布)
```

## 环境本质：Python 3.10 + ROS2

**r1lite 运行环境 = Python 3.10 + ROS2。**

- 客户端的 7 个 pip 依赖（`numpy / opencv-python / msgpack / websockets / typing-extensions / toml / loguru`）声明在本目录 `pyproject.toml`。
- 仓库根目录的 G05 环境执行 `uv sync` 后也包含 `websockets`。
- `rclpy` **不是 pip 包**，靠 `source /opt/ros/humble/setup.bash` 注入。

推荐直接复用仓库根目录的 G05 venv：

```bash
source /opt/ros/humble/setup.bash          # 注入 rclpy
source /path/to/g05-repo/.venv/bin/activate
cd experiments/r1lite && python run.py --config config.toml
```

如只运行客户端单元测试，也可以在本目录使用 `pyproject.toml` 创建轻量环境；部署脚本 `start.sh` 默认使用仓库根目录 `.venv`。

---

## Client ↔ Server 接口契约

```
 Client (R1_LITE)                       Server (serve_policy.py)
 ─────────────────                       ────────────────────────
 ROS2 gather_obs() → raw obs
   images: {cam: ndarray CHW uint8}
   state:  {part: ndarray 1D float32}
   task:   str
   embodiment_type: str (mixture 时)
          ──── msgpack over WebSocket ──►  receive raw obs
                                           build_obs_dict(obs, processor)
                                             ├─ 补 dummy action zeros
                                             ├─ 补 is_pad masks
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

### Client 发送 (raw obs)

| Key | Type | Shape | 说明 |
|-----|------|-------|------|
| `images` | `dict[str, ndarray]` | `{cam_name: (C, H, W) uint8}` | 各相机 RGB 图像（CHW 格式） |
| `state` | `dict[str, ndarray]` | `{part_name: (D,) float32}` | 各部位关节状态 |
| `task` | `str` | — | 自然语言指令（由 InstructionManager 注入） |
| `embodiment_type` | `str` (optional) | — | 多 embodiment mixture 时指定 |

### Client 接收 (raw action)

| Key | Type | Shape | 说明 |
|-----|------|-------|------|
| `{body_part}` | `ndarray` | `(T, D) float32` | 各部位 T 步动作，原始物理量（弧度/米） |

Server 可额外返回 `server_timing` 等 metadata，client 忽略即可。

### 关键原则

1. **Client 不做 normalize / denormalize** — 归一化、坐标变换、合并/拆分全在 server 端
2. **Client 不加载 processor / model config** — 不依赖 server 端 G05 模型包、`omegaconf`、`torch`
3. **Client 不加 batch 维** — `gather_obs()` 返回单帧，server 自行 `unsqueeze(0)`
4. **序列化协议** — msgpack + numpy 自定义编码（见 `utils/websocket/msgpack.py`）

---

## 数据流

```
instruction.txt ──► InstructionManager
                          │
                          ▼
ROS2 Topics ──► Ros2Bridge.gather_obs() ──► obs dict (numpy)
                                                │
                          InstructionManager.get_instruction(obs)
                          注入 obs["task"]      │
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

## 目录结构

```
r1lite/
├── run.py                              # 入口：加载 config.toml → Scheduler.run()
├── start.sh                            # 环境配置（ROS2 + 仓库根目录 venv）
├── config.toml                         # 默认配置（port 8080）
├── config_8082.toml                    # 备用配置（port 8082）
├── pyproject.toml                      # 客户端依赖声明
│
├── scheduler/
│   ├── scheduler.py                    # 主循环：gather → instruct → predict → step
│   └── instruction/
│       ├── instruction.py              # InstructionManager（文件/VLM 指令源）
│       └── instruction.txt             # 当前指令文件
│
├── core/
│   ├── communication/
│   │   ├── ros2_bridge.py              # ROS2 桥接：订阅传感器、发布动作、gather_obs()
│   │   ├── robot_topics.py             # ROS2 topic 名称和 QoS 配置
│   │   └── message_queue.py            # 线程安全消息缓冲区
│   └── inference/
│       ├── inference_engine.py         # 抽象基类（connect / predict_action）
│       ├── websocket_engine.py         # WebSocket 客户端实现
│       └── factory.py                  # create_inference_engine()
│
├── utils/
│   ├── websocket/
│   │   └── msgpack.py                  # numpy ↔ msgpack 序列化
│   ├── message/
│   │   ├── datatype.py                 # RobotAction / Trajectory 数据类
│   │   ├── message_convert.py          # ROS2 msg ↔ numpy 转换、actions_dict_to_trajectory
│   │   └── bbox_utils.py               # bbox 裁剪/格式转换（VLM 模式使用）
│   ├── viz.py                          # CoT bbox 可视化（--visualize）
│   └── episode_recorder.py             # episode 录制（--record）
│
├── scripts/
│   ├── launch_client_8080.sh           # 启动 client（端口 8080）
│   ├── launch_client_8082.sh           # 启动 client（端口 8082）
│   ├── run.sh                          # 简化启动脚本
│   └── record.sh                       # 录制 ROS2 bag
│
└── tests/                              # pytest 单元测试（23 个，本地即可跑）
    ├── conftest.py                     # PYTHONPATH + ROS2/websockets stubs
    ├── test_msgpack.py
    ├── test_factory.py
    ├── test_datatype.py
    └── test_instruction_manager.py
```

> 注：推理服务由仓库根目录的 `scripts/serve_policy.py` 统一提供，client 端不重复携带。

---

## Quick Start

### 1. 安装 uv（如果没有）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装 Python 依赖

```bash
cd /path/to/g05-repo

# 安装仓库根目录 G05 环境（自动创建 .venv）
uv sync
```

### 3. 安装 ROS2 (一般都有配置)

需要 ROS2 Humble（或兼容版本），包含以下包：

```
rclpy, sensor_msgs, geometry_msgs, builtin_interfaces, std_msgs
```

通常随 `ros-humble-desktop` 一起安装，不需要单独 pip install。参考 [ROS2 安装文档](https://docs.ros.org/en/humble/Installation.html)。

### 4. 配置

编辑 `config.toml`，将 `[websocket]` 的 host/port 指向推理服务器：

```toml
[robot]
hardware = "R1_LITE"

[basic]
control_frequency = 15.0    # 控制频率 (Hz)
step_mode = "sync"          # "sync" | "async"
action_steps = 16           # 每次推理执行的步数

[websocket]
host = "10.0.0.100"         # ← 改成推理服务器 IP
port = 8080                 # ← 改成推理服务器端口

[instruction]
use_vlm = false
```

### 5. 运行

```bash
# 激活环境
source /opt/ros/humble/setup.bash
source /path/to/g05-repo/.venv/bin/activate

# 确保推理服务器已启动（在 server 机器上）
# cd /path/to/g05-repo
# PYTHONPATH="src:${PYTHONPATH:-}" python scripts/serve_policy.py --ckpt_path /path/to/model/checkpoints/model_state_dict.pt ...

# 启动 client
cd experiments/r1lite
python run.py --config config.toml

# 或指定配置文件
python run.py --config /path/to/custom_config.toml
```

### 6. 运行测试（不需要 ROS2 和推理服务器）

```bash
source /path/to/g05-repo/.venv/bin/activate
cd experiments/r1lite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v
```

测试通过 `conftest.py` 自动 stub 掉 ROS2 和 websockets 依赖，本地即可运行。`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 可避免当前 shell 里的 ROS2 pytest 插件被自动加载。

---

## 依赖说明

### 运行时（仓库根目录 `uv sync`）

| 包名 | 用途 |
|------|------|
| `numpy` | obs/action 数据格式 |
| `opencv-python` | 图像解码（ROS2 CompressedImage → RGB numpy） |
| `msgpack` | obs/action 序列化，WebSocket 传输协议 |
| `websockets` | WebSocket 客户端，连接推理服务器 |
| `typing-extensions` | `@override` 装饰器 |
| `toml` | 读取 config.toml |
| `loguru` | 日志 |

### 可选（使用本目录轻量环境时）

| extra | 包名 | 何时需要 |
|-------|------|----------|
| `dev` | `pytest` | 运行测试 |

VLM bbox 数据从 ROS2 `hs/vlm_out2vla` topic 消费；client 不再包含远端 bbox 依赖。

### 不需要的包

以下是 server 端依赖，client 端**完全不需要**：

`torch`, server 端 G05 模型包, `omegaconf`, `accelerate`, `hydra`, `lightning`, `diffusers`, `timm`, `tensorboard`, `h5py`, `zarr` 等
