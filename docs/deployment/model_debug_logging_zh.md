# Model Debug Logging

在 `torch.compile` / DDP / FSDP 环境下，模型 forward 内部的 `print()` 和 `logger.info()` 都**不可靠**。本文档描述一套基于直接文件写入的 model debug log 系统，保证在任何训练配置下都能看到日志。

## 为什么 print / logger 不行

| 场景 | `print()` / `logger.info()` 行为 |
|------|----------------------------------|
| 纯 eager（无 compile） | 正常 |
| `torch.compile(model)`（DDP） | **仅 tracing（首次调用）时执行一次**，后续 compiled 执行不触发 |
| FSDP per-layer compile | compile 范围外的代码正常；compile 范围内同上 |
| 分布式 stdout 重定向 | `print` 可能被 buffer，看不到或延迟 |

`torch.compile` 将 Python 代码 trace 成计算图后，`print`/`logger` 等 side effect 不会被编译进图，后续执行时直接跳过。

## 解决方案：直接文件写入

核心思路：用 `fh.write()` + `fh.flush()` 直接写文件，不经过 Python logging 框架。`torch.compile` 无法优化掉文件 IO 操作。

### 1. 在模型中定义 `_model_log` 函数

在需要加日志的模型文件顶部：

```python
import datetime

def _model_log(fh, msg: str):
    """直接写文件的日志函数，compile-safe。"""
    if fh is None:
        return
    fh.write(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')} {msg}\n")
    fh.flush()
```

### 2. 在模型 `__init__` 中预留文件句柄属性

```python
class YourModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... 其他初始化 ...
        self._model_log_fh = None   # 由 finetune.py 注入
        self._fwd_step = 0          # forward 计数器
```

### 3. 在 forward 的关键路径插入日志

```python
def forward(self, ...):
    fh = self._model_log_fh
    step = self._fwd_step

    # 预处理后
    _model_log(fh,
        f"[FWD] step={step} bs={batch_size} seq_len={seq_len} "
        f"split_index={split_index} valid_labels={num_valid}")

    # 截断发生时
    if total_len > self.max_chunk_token_length:
        _model_log(fh,
            f"[CHUNK] TRUNCATING {total_len} -> {self.max_chunk_token_length}")

    # loss 计算后
    _model_log(fh,
        f"[LOSS] step={step} loss={loss.item():.6f} "
        f"ce={ce_loss.item():.6f} fm={fm_loss.item():.6f}")

    self._fwd_step = step + 1
```

### 4. 在 `finetune.py` 中初始化

在训练循环开始前、`output_dir` 确定后：

```python
# ---- Model debug log ----
_inner_model = unwrap_model(model, use_fsdp)
_model_log_path = output_dir / f"model_debug_rank{accelerator.process_index}.log"
_model_log_path.parent.mkdir(parents=True, exist_ok=True)
_model_log_fh = open(_model_log_path, "a")

# 注入到 policy 和内部 model
_inner_model._model_log_fh = _model_log_fh
if hasattr(_inner_model, "model"):
    _inner_model.model._model_log_fh = _model_log_fh
```

如果模型有更深的嵌套（如 `model.backbone`），按需传递：

```python
if hasattr(_inner_model.model, "vision_tower"):
    _inner_model.model.vision_tower._model_log_fh = _model_log_fh
```

## 当前已接入的日志点

### `GalaxeaJointPolicy`（`galaxea_joint_policy.py`）

| 标签 | 位置 | 记录内容 |
|------|------|----------|
| `[PREPROCESS]` | `preprocess_inputs` | prefix_len, suffix_len, total_len, max_chunk, max_pad |
| `[CHUNK]` | `preprocess_inputs` | 截断触发时的原始长度和目标长度 |
| `[PAD]` | `preprocess_inputs` | padding/截断时的长度变化 |
| `[FWD]` | `forward_train` | step, is_vlm, bs, seq_len, split_index, valid_labels, tensor shapes |
| `[LOSS]` | `forward_train` | 总 loss, 各分项 loss, accuracy |

### `GalaxeaJoint`（`galaxea_joint.py`）

| 标签 | 位置 | 记录内容 |
|------|------|----------|
| `[CE]` | `cal_ce_acc` | valid_tokens, ce_loss, accuracy |
| `[CE] WARNING` | `cal_ce_acc` | 所有 labels 为 -100 时的警告 |
| `[MODEL]` | `forward` 末尾 | 执行路径(joint/continuous_only), fm_loss, ce_loss, accuracy |

## 输出文件

每个 rank 独立写入 `<output_dir>/model_debug_rank{rank}.log`。

```
14:22:31.123456 [PREPROCESS] prefix_len=742 suffix_len=38 total_len=780 max_chunk=1300 max_pad=1300
14:22:31.123789 [PAD] PADDING 780 -> 1300 (pad_len=520)
14:22:31.234567 [FWD] step=0 is_vlm=False bs=4 seq_len=1300 split_index=742 valid_labels=152 pixel=(4, 2, 3, 224, 224) action=(4, 8, 14)
14:22:32.345678 [CE] valid_tokens=152 ce_loss=8.234567 acc=0.0132
14:22:32.456789 [MODEL] path=joint fm_loss=0.123456 ce_loss=8.234567 acc=0.013... continuous_action=True
14:22:32.567890 [LOSS] step=0 loss=8.358023 fm_loss=0.123456 ce_loss=8.234567 acc=0.013...
```

## 给新模型加日志的步骤

1. 在模型文件顶部复制 `_model_log` 函数
2. 在 `__init__` 中加 `self._model_log_fh = None`
3. 在 forward 的每个分支/关键点加 `_model_log(self._model_log_fh, f"[TAG] ...")`
4. 在 `finetune.py` 的日志初始化块中把 `_model_log_fh` 注入到新模型

## 注意事项

- **不要用 `logger.info()` 替代**：在 compile 范围内不可靠
- **每行都 `flush()`**：保证崩溃前的日志不丢失
- **每个 rank 独立文件**：避免多进程写同一文件导致乱序
- **`.item()` 调用会触发 GPU 同步**：生产环境如果影响性能，可以只在前 N 步或每 K 步记录
- **`mode="a"` 追加模式**：resume 训练时日志不会被覆盖

---
*最后修改: 2026.03.09*
