# SO100 部署

把训练好的 G05 checkpoint 部署到 SO100/SO101 机械臂。采用 **server / client 分离**架构,通过 WebSocket 通信:

- **server**:跑在带 GPU 的机器上,加载 checkpoint 做推理,用本仓库的 `g05` 训练环境,直接复用通用的 `scripts/serve_policy.py`(无 SO100 专用 server)。
- **client**:跑在连着机械臂的本地机上,负责读相机 / 读关节 / 下发动作,用独立的 `lerobot` conda 环境(不依赖 `g05` 包)。

```mermaid
flowchart LR
    subgraph GPU 机器 (g05 训练环境)
        S[scripts/serve_policy.py<br/>加载 ckpt + 推理]
    end
    subgraph 机器人本地机 (lerobot 环境)
        C[so100_policy_client.py<br/>相机/关节/下发动作]
        R[(SO100 机械臂<br/>+ 相机)]
    end
    C -- obs: images+state --> S
    S -- action chunk --> C
    C <--> R
```

## 文件说明

| 文件 | 跑在哪 | 说明 |
|------|--------|------|
| `so100_policy_client.py` | 本地机 | 机器人 client。单后台线程独占串口总线 + 每相机独立读帧线程 + action_fps Hz 推理循环。**缺失相机(如 wrist_left / head)由 client 全零黑帧 padding** 后再发给 server(并能从 server 的相机契约错误自动学习要补哪些 slot)。仅依赖 `lerobot` 和本仓库 `scripts/utils/{policy_ws_client,mem_live_viz}.py` |
| `debug_action_scale.py` | 本地机 | 标定工具:检查 lerobot SOFollower 返回的关节单位 / action scale |
| `client_config.yaml` | 本地机 | client 的 proprio OOD guard 配置(clip / zscore,防止越界关节态拖垮预测) |
| `environment.yml` | 本地机 | client 的 conda 环境(`lerobot==0.5.2` + `feetech-servo-sdk` + `pyserial` 等) |
| `start_server.sh` / `start_client.sh` | 各自 | 启动脚本,按需改相机映射 / 端口 / checkpoint 路径 |

## 快速开始

### 1. server（GPU 机,g05 环境）

```bash
source .venv/bin/activate
bash experiments/so100/start_server.sh /path/to/checkpoint.pt
```

### 2. client（本地机,lerobot 环境）

```bash
# 首次:创建 lerobot 环境(已含 pip lerobot==0.5.2,无需 vendor 源码)
conda env create -f experiments/so100/environment.yml
conda activate lerobot

# 改 start_client.sh 里的 --camera-index / --camera-map 匹配你的相机,然后:
bash experiments/so100/start_client.sh
```

> **关于 lerobot**:client 用官方未改动的 `lerobot`,直接 pip 安装(见 `environment.yml`),仓库不再 vendor 其源码。

## 训练-部署一致性要点

- SO100 实机只有 **exterior + wrist_right** 两路相机;模型按多相机训练,**client** 会把缺失相机(如 wrist_left / head)**全零 padding** 后再发给通用 server,与训练时的 camera-drop 增强对齐。
- client `--camera-map` 必须把实机相机名映射到模型期望的 slot(`exterior`、`wrist_right`)。
- 关节坐标系换算见 `so100_policy_client.py` 头部 docstring(`signs` / `offsets`)。
