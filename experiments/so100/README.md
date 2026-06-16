# SO100 Deployment

Deploy a trained G05 checkpoint to SO100/SO101 robot arms. The setup uses a **server / client split** over WebSocket:

- **server**: runs on a GPU machine, loads the checkpoint, and performs inference with this repository's `g05` training environment. It reuses the generic `scripts/serve_policy.py`; there is no SO100-specific server.
- **client**: runs on the local machine connected to the robot arm. It reads cameras and joints, sends observations, and executes returned actions. It uses an independent `lerobot` conda environment and does not depend on the `g05` package.

```mermaid
flowchart LR
    subgraph GPU machine (g05 training environment)
        S[scripts/serve_policy.py<br/>load checkpoint + infer]
    end
    subgraph local robot machine (lerobot environment)
        C[so100_policy_client.py<br/>cameras/joints/action execution]
        R[(SO100 arm<br/>+ cameras)]
    end
    C -- obs: images+state --> S
    S -- action chunk --> C
    C <--> R
```

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `so100_policy_client.py` | local machine | Robot client. One background thread owns the serial bus, each camera has its own frame reader thread, and the inference loop runs at `action_fps`. Missing cameras such as `wrist_left` or `head` are zero-padded by the client before sending to the server. The client can learn missing required camera slots from server camera-contract errors. It depends only on `lerobot` and this repository's `scripts/utils/{policy_ws_client,mem_live_viz}.py`. |
| `debug_action_scale.py` | local machine | Calibration tool for checking SOFollower joint units and action scale from lerobot. |
| `client_config.yaml` | local machine | Client proprio OOD guard configuration: clipping and z-score checks that prevent out-of-range joint states from damaging predictions. |
| `environment.yml` | local machine | Conda environment for the client: `lerobot==0.5.2`, `feetech-servo-sdk`, `pyserial`, and related dependencies. |
| `start_server.sh` / `start_client.sh` | respective machines | Startup scripts; edit camera mapping, port, and checkpoint path as needed. |

## Quick Start

### 1. Server: GPU Machine With G05 Environment

```bash
source .venv/bin/activate
bash experiments/so100/start_server.sh /path/to/checkpoint.pt
```

### 2. Client: Local Machine With LeroBot Environment

```bash
# First run: create the lerobot environment. It already includes pip lerobot==0.5.2; no vendored source is required.
conda env create -f experiments/so100/environment.yml
conda activate lerobot

# Edit --camera-index / --camera-map in start_client.sh to match your cameras, then run:
bash experiments/so100/start_client.sh
```

About lerobot: the client uses the official unmodified `lerobot` installed directly with pip; see `environment.yml`. This repository no longer vendors lerobot source code.

## Training And Deployment Consistency

- The physical SO100 setup has only **exterior + wrist_right** cameras. The model may be trained with more cameras, so the **client** zero-pads missing cameras such as `wrist_left` or `head` before sending observations to the generic server. This aligns with camera-drop augmentation during training.
- Client `--camera-map` must map physical camera names to the model's expected slots, such as `exterior` and `wrist_right`.
- Joint coordinate conversion is documented in the top-level docstring of `so100_policy_client.py`, especially `signs` and `offsets`.
