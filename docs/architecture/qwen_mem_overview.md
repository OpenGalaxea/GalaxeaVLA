# Qwen3.5 MEM Multi-Frame Overview

## What Changed

MEM, the Multi-frame Encoder Module, was implemented in the Qwen3.5 ViT so the model can see roughly the last five seconds of history without increasing the number of VLM tokens.

### Changed Files

| File | Change |
|------|--------|
| `src/g05/models/g05/qwen35/vision.py` | Core factorized/cascaded temporal + spatial attention and token drop. |
| `src/g05/models/g05/g05_model_qwen35.py` | Frame drop, with 30% probability of dropping history, plus patch-order fix. |
| `src/g05/data/base_lerobot_dataset.py` | `obs_size` as int and `obs_stride_second` as seconds, automatically converted to stride steps. |
| `src/g05/data/mixture_lerobot_dataset.py` | Signature alignment and `obs_stride_second` forwarding. |
| `src/g05/data_processor/processor/samples_builder.py` | `hardcode_instruction` support. |
| `scripts/finetune.py` | `use_meta_device_for_ckpt` switch. |
| `configs/model/g05.yaml` | `spacetime_mode` and `token_drop_layer` fields. |
| `configs/data/<task>.yaml` | `obs_size` / `obs_stride_second`; single-frame defaults to `obs_size: 1`, while MEM checkpoint `.hydra/config.yaml` uses 6. |

## Flow Comparison

### Single Frame

```text
1 frame x 3 cameras -> ViT, 24 layers of spatial attention -> PatchMerger -> 224 tokens -> VLM
```

### MEM Multi-Frame

```text
6 frames x 3 cameras -> ViT, 18 spatial layers + 6 factorized T+S layers -> Token Drop, keep current frame only -> PatchMerger -> 224 tokens -> VLM
```

The VLM is unaware of the change: it still receives 224 tokens in the same format, but each token representation has incorporated six-frame temporal information.

## Core Code Changes

### 1. `vision.py`: Factorized Temporal + Spatial Attention

`Qwen3_5VisionBlock` adds `forward_spacetime()`:

```python
def _spacetime_factorized(self, hidden_states, cu_seqlens, position_embeddings,
                          num_frames, bsz, causal_mask, temporal_pe):
    # 1. Add temporal PE. e(0)=0, so the current frame is unchanged.
    hs = hidden_states + temporal_pe

    # 2. Project QKV once. Shared weights, zero new parameters.
    q, k, v = self.attn._project_qkv(self.norm1(hs))

    # 3. Temporal: per-patch cross-frame causal attention -> V'.
    #    Each patch position attends across 6 frames independently.
    #    V' merges same-position information from all 6 frames.
    #    out_proj is not applied here.

    # 4. Spatial: use original Q,K plus 2D RoPE and V' as values.
    #    out_proj is applied only once.

    # Result: residual + spatial_out
```

Why factorized:

- With K=1 it is exactly equivalent to the original ViT: `e(0)=0`, `V'=V`, and temporal attention is identity.
- With 30% frame drop, the K=1 path does not conflict with the K>1 training target.
- Cascaded attention is not equivalent because spatial attention would re-project QKV, making `Q_new = q_proj(v_proj(x))` instead of `q_proj(x)`.

`Qwen3_5VisionModel.forward()` has two paths:

```python
if num_frames <= 1 or self.temporal_freq <= 0:
    # Single-frame path: zero overhead and identical to the original model.
    for blk in self.blocks:
        blk(hidden_states, cu_seqlens, position_embeddings)
else:
    # Multi-frame MEM path.
    for i, blk in enumerate(self.blocks):
        if use_temporal:
            blk.forward_spacetime(...)  # T+S layer
        else:
            blk(...)  # spatial-only layer

        if i == drop_idx:  # Token Drop
            hidden_states = hidden_states[:, -1]  # keep current frame only
            # recompute cu_seqlens and position_embeddings
```

### 2. `g05_model_qwen35.py`: Frame Drop And Patch Order

Frame Drop drops all history frames with 30% probability:

```python
# Important: broadcast keeps all DDP ranks identical, preventing NCCL hangs.
drop_flag = torch.rand(1, device=device) < frame_drop_prob
torch.distributed.broadcast(drop_flag, src=0)

if drop_history:
    cam_tensor = cam_tensor[:, -1:]  # keep current frame only
```

Patch-order fix:

```python
# Before, row-major: PatchMerger merged four patches from the same row.
# After, merge-grouped: PatchMerger merges a 2x2 spatial neighborhood.
patches_k = images_temporal.reshape(
    total_k, tps, C, merged_h, merge_size, ps, merged_w, merge_size, ps
).permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(...)
```

### 3. Config Fields

```yaml
# configs/model/g05.yaml
vision:
  temporal_freq: 0           # 0 disables MEM; 4 means one T+S layer every 4 layers
  spacetime_mode: factorized # "factorized" or "cascaded"
  token_drop_layer: null     # null means depth, drop at the end

# MEM task config
model.model_arch:
  mem_frame_drop_prob: 0.3
  checkpoint_vision: true
  cond_steps: 1              # VLM sees only 1 frame after MEM compression

data:
  obs_size: 6
  obs_stride_second: 1.0     # one second between frames, independent of fps
```

### 4. `obs_stride_second`

Motivation: different embodiments have different data fps, such as r1lite at 15fps and agibot at 30fps, but the history window should be consistent in real time.

```yaml
data:
  obs_size: 6                # number of frames, shared by image and state
  obs_stride_second: 1.0     # seconds between frames; 0 or omitted means consecutive frames
```

Dataset logic:

```python
data_fps = meta.fps

if obs_stride_second > 0:
    obs_stride = max(1, round(obs_stride_second * data_fps))
else:
    obs_stride = 1

# offsets = [-(obs_size-1)*obs_stride/fps, ..., -obs_stride/fps, 0]
```

Effects:

| Data fps | obs_stride_second | stride_steps | history_window |
|----------|-------------------|--------------|----------------|
| 15, r1lite | 1.0 | 15 | 5.0s |
| 30, agibot | 1.0 | 30 | 5.0s |
| 15 | 0.5 | 8 | 2.8s |
| 30 | 2.0 | 60 | 10.0s |

Backward compatibility: old configs only need `obs_size: 1` and can omit `obs_stride_second`. This is equivalent to consecutive single-frame input.

Training info logs can be used for validation:

```text
[obs] .../R1_Lite/...: obs_size=6, obs_stride_second=1.0s -> stride_steps=15 (fps=15), history_window=5.0s
[obs] .../agibot/...:  obs_size=6, obs_stride_second=1.0s -> stride_steps=30 (fps=30), history_window=5.0s
```

## Temporal + Spatial Layer Schedule

For `temporal_freq=4`, `token_drop_layer=null`, and `depth=24`:

```text
T+S layers: {3, 7, 11, 15, 19, 23}  (6 layers)
S-only:     {0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22}  (18 layers)
Token Drop: after layer 23
```

## Benchmark

DDP 2xH20, batch size 4 per GPU, 3 embodiments: r1lite, r1pro, agibot.

| Mode | ms/step | samples/sec | MEM slowdown |
|------|---------|-------------|--------------|
| single-frame | 1300 | 3.08 sps | - |
| MEM K=6 | 1925 | 2.08 sps | **1.5x** |

## K=1 Equivalence

```text
factorized T+S with K=1:
  e(0) = 0              -> temporal PE is zero
  softmax([scalar]) = 1 -> V' = V, temporal attention is identity
  spatial uses original Q,K

Result: max diff = 0.00 in code validation.
```

This guarantees that with 30% frame drop, where K=1, the model sees the exact pretrained ViT rather than a conflicting training objective.

## Known Issues

1. **The VLM has no explicit temporal signal**: after token drop, MRoPE temporal `position_id` is constant, so the VLM cannot directly distinguish "history present" from "no history".
2. **Qwen3.5 ViT 2D RoPE**: RoPE encodes only row and column; temporal information is injected as sinusoidal PE at the input level.
3. **Patch order**: the 2026-04-29 C/tps reshape is numerically equivalent in VLA single-frame scenarios because expand duplicates frames, but its semantics are less accurate than the original HF order.

## Multi-Frame Inference Input: Training Format vs Real-Robot Deployment

### Training Format

Training data is stored as **MP4 (H264)**. The DataLoader decodes frames at selected timestamps through pyav/torchcodec into RGB tensors. The model sees decoded pixels and is unaware of the storage codec.

### Real-Robot Inference Needs No Video Codec

At inference time, cameras provide raw RGB frames directly. A ring buffer is enough:

```python
frame_buffer = deque(maxlen=6)
while running:
    rgb = camera.read()          # raw RGB, about 0ms
    frame_buffer.append(rgb)     # memcpy about 0.02ms
    input = stack(frame_buffer)  # [6, 3, H, W]
    action = model(input)        # ViT forward, about 1.5x single-frame
```

Additional codec latency is **0 ms** compared with single-frame inference. The only extra cost is that ViT processes 18 images instead of 3 images.

### If Encoding/Decoding Is Forced

Test setup: 720x1280, CPU software codec, 6 frames at 1-second stride.

Encoding latency:

| Frames | H264 encode | H265 encode |
|--------|-------------|-------------|
| 1 frame | 13.5 ms | about 30 ms |
| 6 frames, MEM | 64 ms | about 150-300 ms |
| 24 consecutive frames | 273 ms | 600+ ms |

H265 encoding is 2-5x slower than H264 because of more complex block partitioning and motion search; it gives about 30-50% smaller files at the same quality.

Decode latency with seeking, 6 frames at 1-second stride:

| GOP size | GOPs crossed | Farthest P-frame distance | Decode time | Per-frame average |
|----------|--------------|---------------------------|-------------|-------------------|
| 5 | 6 | 0, all I-frames | 112 ms | 18.6 ms |
| 15 | 6 | 0, exactly I-frames | 102 ms | 17.1 ms |
| 30 | 3 | 15 frames | 105 ms | 17.4 ms |
| 60 | 2 | 45 frames | 105 ms | 17.4 ms |
| 90 | 1 | 75 frames | 100 ms | 16.6 ms |

GOP relationship:

```text
H264 GOP, GOP=15, fps=15:

GOP 0            GOP 1            GOP 2            GOP 3            GOP 4            GOP 5
I P P ... P      I P P ... P      I P P ... P      I P P ... P      I P P ... P      I P P ...
0 1 2 ... 14     15 16 ... 29     30 31 ... 44     45 46 ... 59     60 61 ... 74     75 ...
^                ^                ^                ^                ^                ^
take frame 0     take frame 15    take frame 30    take frame 45    take frame 60    take frame 75
```

- **I-frame**: complete encoding, independently decodable.
- **P-frame**: stores difference from previous frames and must be decoded from the nearest I-frame forward.

The cost to decode a frame is seek to the previous I-frame plus sequential P-frame decode until the target frame. Larger P-frame distance requires more intermediate frames.

Measured conclusion: GOP size has little impact on the MEM six-frame scenario, under 10%, because the bottleneck is fixed container open + seek I/O, about 17ms per frame, not P-frame decoding itself.

Total added latency:

| Method | Encode | Decode | Total |
|--------|--------|--------|-------|
| H264 software | 64 ms | 100 ms | about 164 ms |
| H265 software | 150-300 ms | 110-130 ms | about 260-430 ms |
| H264 NVENC hardware | 5-10 ms | 3-8 ms | about 8-18 ms |
| Raw frames directly | 0 | 0 | 0 ms |

### Conclusion

Inference does **not** need video encoding or decoding. The model input is an RGB tensor and does not care whether pixels were stored through H264. H264 during training is a storage-format choice; its compression artifacts are negligible at the visible PSNR level.

## Run Command

```bash
source startg05.sh
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export G05_OUTPUT_DIR=$(pwd)/runs

# MEM fine-tuning example with LIBERO.
# MEM models need obs_size=6 to match checkpoint training.
bash scripts/run/finetune.sh 2 libero \
  --test model.max_steps=30 model.max_epochs=null model.batch_size=4
```
