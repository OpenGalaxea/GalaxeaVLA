# Qwen3.5 MEM 多帧改动概述

## 改了什么

在 Qwen3.5 ViT 中实现了 MEM（Multi-frame Encoder Module），让模型能看到过去 5 秒的历史帧，同时 VLM 的 token 数不增加。

### 改动文件

| 文件 | 改动 |
|------|------|
| `src/g05/models/g05/qwen35/vision.py` | 核心：factorized/cascaded T+S attention + token drop |
| `src/g05/models/g05/g05_model_qwen35.py` | frame drop (30% 概率丢弃历史帧) + patch 顺序修复 |
| `src/g05/data/base_lerobot_dataset.py` | obs_size (int) + obs_stride_second (float秒) → 自动算 stride_steps |
| `src/g05/data/mixture_lerobot_dataset.py` | 签名对齐，传递 obs_stride_second |
| `src/g05/data_processor/processor/samples_builder.py` | hardcode_instruction 支持 |
| `scripts/finetune.py` | use_meta_device_for_ckpt 开关 |
| `configs/model/g05.yaml` | spacetime_mode / token_drop_layer 字段 |
| `configs/data/<task>.yaml` | `obs_size` / `obs_stride_second`（单帧默认 `obs_size: 1`，MEM checkpoint 的 `.hydra/config.yaml` 中为 6） |

---

## 流程对比

### 单帧（原版）

```
1 帧 × 3 cameras → ViT (24层 spatial attention) → PatchMerger → 224 tokens → VLM
```

### MEM 多帧

```
6 帧 × 3 cameras → ViT (18层 spatial + 6层 T+S factorized) → Token Drop (只保留当前帧) → PatchMerger → 224 tokens → VLM
```

**VLM 完全无感知**——同样 224 tokens、同样格式，但每个 token 的 representation 已融入 6 帧时序信息。

---

## 核心代码改动详解

### 1. vision.py — Factorized T+S Attention

**Qwen3_5VisionBlock** 新增 `forward_spacetime()`:

```python
def _spacetime_factorized(self, hidden_states, cu_seqlens, position_embeddings,
                          num_frames, bsz, causal_mask, temporal_pe):
    # 1. 加 temporal PE (e(0)=0, 当前帧不受影响)
    hs = hidden_states + temporal_pe

    # 2. QKV 一次投影（共享权重，零新参数）
    q, k, v = self.attn._project_qkv(self.norm1(hs))

    # 3. Temporal: per-patch 跨帧 causal attention → V'
    #    每个 patch 位置独立地在 6 帧间做 attention
    #    V' 融合了同位置 6 帧的信息
    #    不调 out_proj

    # 4. Spatial: 原始 Q,K + 2D RoPE, 用 V' 作为 values
    #    out_proj 只调一次

    # 结果: residual + spatial_out
```

**为什么用 factorized**:
- K=1 时严格等价于原版 ViT（e(0)=0 → V'=V → 和不做 temporal 完全一样）
- 30% frame drop 时走 K=1 路径，不会和 K>1 的训练目标冲突
- cascaded 不等价（spatial 会重新 QKV 投影，Q_new = q_proj(v_proj(x)) ≠ q_proj(x)）

**Qwen3_5VisionModel.forward()** 双路径:

```python
if num_frames <= 1 or self.temporal_freq <= 0:
    # 单帧路径（零开销，和原版完全一样）
    for blk in self.blocks:
        blk(hidden_states, cu_seqlens, position_embeddings)
else:
    # 多帧 MEM 路径
    for i, blk in enumerate(self.blocks):
        if use_temporal:
            blk.forward_spacetime(...)  # T+S 层
        else:
            blk(...)  # S-only 层

        if i == drop_idx:  # Token Drop
            hidden_states = hidden_states[:, -1]  # 只保留当前帧
            # 重算 cu_seqlens 和 position_embeddings
```

### 2. g05_model_qwen35.py — Frame Drop + Patch 顺序

**Frame Drop** (30% 概率丢弃所有历史帧):

```python
# 关键: broadcast 保证所有 DDP ranks 一致（否则 NCCL 挂死）
drop_flag = torch.rand(1, device=device) < frame_drop_prob
torch.distributed.broadcast(drop_flag, src=0)

if drop_history:
    cam_tensor = cam_tensor[:, -1:]  # 只保留当前帧
```

**Patch 顺序修复** (merge-grouped):

```python
# 修复前 (row-major): PatchMerger 合并同行 4 个 patch ✗
# 修复后 (merge-grouped): PatchMerger 合并 2×2 空间邻域 ✓
patches_k = images_temporal.reshape(
    total_k, tps, C, merged_h, merge_size, ps, merged_w, merge_size, ps
).permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(...)
```

### 3. Config 字段

```yaml
# configs/model/g05.yaml
vision:
  temporal_freq: 0          # 0=禁用, 4=每4层一个T+S
  spacetime_mode: factorized # "factorized" 或 "cascaded"
  token_drop_layer: null     # null=depth(最后才drop)

# MEM task config
model.model_arch:
  mem_frame_drop_prob: 0.3   # 30% 丢弃历史帧
  checkpoint_vision: true    # per-layer gradient checkpointing
  cond_steps: 1              # VLM 只看 1 帧

data:
  obs_size: 6               # 取 6 帧历史 (image + state 共用)
  obs_stride_second: 1.0    # 每帧间隔 1 秒 (自动适配任意 fps)
```

### 4. obs_stride_second 设计

**动机**: 不同 embodiment 的数据 fps 不同（r1lite=15fps, agibot=30fps），但希望历史窗口在**时间尺度上一致**。

**schema（标量）**:

```yaml
data:
  obs_size: 6                # int — 取多少帧 (image 和 state 相同)
  obs_stride_second: 1.0     # float — 帧间隔(秒), 0 或不设 = 连续帧
```

**代码逻辑** (`base_lerobot_dataset.py`):

```python
# 读取数据集实际 fps（从 meta/info.json）
data_fps = meta.fps  # e.g. 15 or 30

# 自动将秒转为步数
if obs_stride_second > 0:
    obs_stride = max(1, round(obs_stride_second * data_fps))
else:
    obs_stride = 1

# 构造 delta_timestamps
# offsets = [-(obs_size-1)*obs_stride/fps, ..., -obs_stride/fps, 0]
```

**效果**:

| 数据 fps | obs_stride_second | stride_steps | history_window |
|----------|-------------------|--------------|----------------|
| 15 (r1lite) | 1.0 | 15 | 5.0s |
| 30 (agibot) | 1.0 | 30 | 5.0s |
| 15 | 0.5 | 8 | 2.8s |
| 30 | 2.0 | 60 | 10.0s |

**向后兼容**: 旧 config 只需 `obs_size: 1`（不写 `obs_stride_second`），等价于连续单帧，无需任何修改。

**训练时 info log** (可用于验证):
```
[obs] .../R1_Lite/...: obs_size=6, obs_stride_second=1.0s → stride_steps=15 (fps=15), history_window=5.0s
[obs] .../agibot/...:  obs_size=6, obs_stride_second=1.0s → stride_steps=30 (fps=30), history_window=5.0s
```

---

## T+S 层 Schedule

Config: `temporal_freq=4, token_drop_layer=null(=24), depth=24`

```
T+S layers: {3, 7, 11, 15, 19, 23}  (6 层)
S-only:     {0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22}  (18 层)
Token Drop: layer 23 之后
```

---

## Benchmark

DDP 2×H20, bsz=4/GPU, 3 embodiments (r1lite + r1pro + agibot):

| 模式 | ms/step | samples/sec | MEM 慢 |
|------|---------|-------------|--------|
| 单帧 | 1300 | 3.08 sps | — |
| MEM K=6 | 1925 | 2.08 sps | **1.5x** |

---

## K=1 等价性

```
factorized T+S with K=1:
  e(0) = 0           → temporal PE 为零，不影响 QKV
  softmax([scalar])=1 → V' = V (temporal attn 是 identity)
  spatial 用原始 Q,K  → 和不做 temporal 完全一样

结果: max diff = 0.00 (代码验证)
```

这保证了 30% frame drop (K=1) 时，模型看到的是**完全相同的 pretrained ViT**，不会产生冲突的训练目标。

---

## 已知问题

1. **VLM 没有时序信息**: token drop 后 MRoPE temporal position_id 是常数，VLM 无法区分"有历史"和"没历史"
2. **Qwen3.5 ViT 2D RoPE**: 只编码 (row, col)，temporal 由 sinusoidal PE 在 input 层面注入
3. **patch 顺序**: 0429 版本的 C/tps reshape 在 VLA 单帧场景下数值等价（因为 expand 复制），但语义上不如 HF 原版准确

---

## 推理时的多帧输入：训练格式 vs 真机部署

### 训练时的数据格式

训练数据以 **MP4 (H264)** 存储，DataLoader 通过 pyav/torchcodec 解码指定 timestamp 的帧 → RGB tensor。模型看到的是**解码后的纯像素**，不感知编码格式。

### 真机推理：不需要视频编解码

推理时相机直接给 raw RGB 帧，维护一个 ring buffer 即可：

```python
frame_buffer = deque(maxlen=6)  # 6帧 ring buffer
while running:
    rgb = camera.read()          # raw RGB, ~0ms
    frame_buffer.append(rgb)     # memcpy ~0.02ms
    input = stack(frame_buffer)  # [6, 3, H, W]
    action = model(input)        # ViT forward (1.5x 单帧)
```

**额外延迟 = 0 ms**（相对单帧，唯一差异是 ViT 处理 18 张图而非 3 张）。

### 如果强制加编解码会怎样？

测试环境：720×1280 (720p)，CPU software codec，6帧间隔1s。

#### 编码延迟

| 帧数 | H264 编码 | H265 编码 |
|------|-----------|-----------|
| 1帧 | 13.5 ms | ~30 ms |
| 6帧 (MEM) | 64 ms | ~150-300 ms |
| 24帧连续 | 273 ms | ~600+ ms |

H265 编码比 H264 **慢 2-5x**（更复杂的块划分和运动搜索），换来的是更好的压缩率（同质量体积小 30-50%）。

#### 解码延迟 (seek 方式，6帧@1s stride)

| GOP size | 跨越 GOP 数 | 最远 P-frame 距离 | 解码耗时 | 每帧平均 |
|----------|------------|-----------------|---------|---------|
| 5 | 6 | 0 (全 I-frame) | 112 ms | 18.6 ms |
| 15 | 6 | 0 (恰好命中 I-frame) | 102 ms | 17.1 ms |
| 30 | 3 | 15 帧 | 105 ms | 17.4 ms |
| 60 | 2 | 45 帧 | 105 ms | 17.4 ms |
| 90 | 1 | 75 帧 | 100 ms | 16.6 ms |

#### GOP 与解码的关系

```
H264 GOP 结构 (GOP=15, fps=15):

GOP 0            GOP 1            GOP 2            GOP 3            GOP 4            GOP 5
I P P ... P      I P P ... P      I P P ... P      I P P ... P      I P P ... P      I P P ...
0 1 2 ... 14     15 16 ... 29     30 31 ... 44     45 46 ... 59     60 61 ... 74     75 ...
↑                ↑                ↑                ↑                ↑                ↑
取帧0            取帧15           取帧30           取帧45           取帧60           取帧75
```

- **I-frame（关键帧）**：完整编码，可独立解码
- **P-frame（预测帧）**：存与前帧差异，必须从最近 I-frame 顺序解码

**解码某一帧的代价** = seek 到前一个 I-frame + 顺序解 P-frame 直到目标帧。P-frame 距离越远，需要解码的中间帧越多。

**实测结论**：GOP 大小对 MEM 6帧场景影响很小（<10%），因为瓶颈是 container open + seek 的固定 I/O 开销（~17ms/帧），而非 P-frame 解码本身。

#### 总额外延迟

| 方案 | 编码 | 解码 | 总延迟 |
|------|------|------|--------|
| H264 软件 | 64 ms | 100 ms | **~164 ms** |
| H265 软件 | 150-300 ms | 110-130 ms | **~260-430 ms** |
| H264 NVENC 硬件 | 5-10 ms | 3-8 ms | **~8-18 ms** |
| 直接用 raw frame | 0 | 0 | **0 ms** |

### 结论

推理时**完全不需要视频编解码**。模型输入是 RGB tensor，不关心像素是否经过 H264 压缩。训练时的 H264 只是存储格式的选择，对模型表征无实质影响（压缩伪影在 PSNR 层面人眼不可见）。

---

## 运行命令

```bash
source startg05.sh
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export G05_OUTPUT_DIR=$(pwd)/runs

# MEM finetune（以 libero 为例；MEM 模型需 obs_size=6 与 checkpoint 训练时一致）
bash scripts/run/finetune.sh 2 libero \
  --test model.max_steps=30 model.max_epochs=null model.batch_size=4
```
