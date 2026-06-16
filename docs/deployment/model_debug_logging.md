# Model Debug Logging

Inside model `forward()` calls, `print()` and `logger.info()` are unreliable under `torch.compile`, DDP, and FSDP. This document describes a model debug logging system based on direct file writes so logs remain visible under any training configuration.

## Why Print And Logger Are Not Enough

| Scenario | `print()` / `logger.info()` behavior |
|----------|--------------------------------------|
| Pure eager mode without compile | Works normally. |
| `torch.compile(model)` with DDP | Runs only once during tracing, usually on the first call; compiled executions do not trigger it. |
| FSDP per-layer compile | Code outside the compile region works normally; code inside behaves like compiled DDP. |
| Distributed stdout redirection | `print` can be buffered, delayed, or invisible. |

After `torch.compile` traces Python code into a graph, side effects such as `print` and logger calls are not compiled into the graph and are skipped on later executions.

## Solution: Direct File Writes

The core idea is to write directly with `fh.write()` and `fh.flush()` without going through the Python logging framework. `torch.compile` cannot optimize away file I/O.

### 1. Define `_model_log` In The Model File

Add this near the top of the model file that needs logging:

```python
import datetime

def _model_log(fh, msg: str):
    """Compile-safe direct file logging."""
    if fh is None:
        return
    fh.write(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')} {msg}\n")
    fh.flush()
```

### 2. Reserve File Handle Attributes In `__init__`

```python
class YourModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... other initialization ...
        self._model_log_fh = None   # injected by finetune.py
        self._fwd_step = 0          # forward counter
```

### 3. Add Logs At Key Points In `forward`

```python
def forward(self, ...):
    fh = self._model_log_fh
    step = self._fwd_step

    # After preprocessing.
    _model_log(fh,
        f"[FWD] step={step} bs={batch_size} seq_len={seq_len} "
        f"split_index={split_index} valid_labels={num_valid}")

    # When truncation happens.
    if total_len > self.max_chunk_token_length:
        _model_log(fh,
            f"[CHUNK] TRUNCATING {total_len} -> {self.max_chunk_token_length}")

    # After loss calculation.
    _model_log(fh,
        f"[LOSS] step={step} loss={loss.item():.6f} "
        f"ce={ce_loss.item():.6f} fm={fm_loss.item():.6f}")

    self._fwd_step = step + 1
```

### 4. Initialize In `finetune.py`

After `output_dir` is known and before the training loop starts:

```python
# ---- Model debug log ----
_inner_model = unwrap_model(model, use_fsdp)
_model_log_path = output_dir / f"model_debug_rank{accelerator.process_index}.log"
_model_log_path.parent.mkdir(parents=True, exist_ok=True)
_model_log_fh = open(_model_log_path, "a")

# Inject into the policy and inner model.
_inner_model._model_log_fh = _model_log_fh
if hasattr(_inner_model, "model"):
    _inner_model.model._model_log_fh = _model_log_fh
```

For deeper nesting, such as `model.backbone`, pass the handle as needed:

```python
if hasattr(_inner_model.model, "vision_tower"):
    _inner_model.model.vision_tower._model_log_fh = _model_log_fh
```

## Existing Log Points

### `GalaxeaJointPolicy` (`galaxea_joint_policy.py`)

| Tag | Location | Contents |
|-----|----------|----------|
| `[PREPROCESS]` | `preprocess_inputs` | prefix_len, suffix_len, total_len, max_chunk, max_pad |
| `[CHUNK]` | `preprocess_inputs` | original and target lengths when truncation is triggered |
| `[PAD]` | `preprocess_inputs` | length changes during padding or truncation |
| `[FWD]` | `forward_train` | step, is_vlm, bs, seq_len, split_index, valid_labels, tensor shapes |
| `[LOSS]` | `forward_train` | total loss, individual loss terms, accuracy |

### `GalaxeaJoint` (`galaxea_joint.py`)

| Tag | Location | Contents |
|-----|----------|----------|
| `[CE]` | `cal_ce_acc` | valid_tokens, ce_loss, accuracy |
| `[CE] WARNING` | `cal_ce_acc` | warning when all labels are -100 |
| `[MODEL]` | end of `forward` | execution path, joint or continuous_only, fm_loss, ce_loss, accuracy |

## Output File

Each rank writes its own `<output_dir>/model_debug_rank{rank}.log`.

```text
14:22:31.123456 [PREPROCESS] prefix_len=742 suffix_len=38 total_len=780 max_chunk=1300 max_pad=1300
14:22:31.123789 [PAD] PADDING 780 -> 1300 (pad_len=520)
14:22:31.234567 [FWD] step=0 is_vlm=False bs=4 seq_len=1300 split_index=742 valid_labels=152 pixel=(4, 2, 3, 224, 224) action=(4, 8, 14)
14:22:32.345678 [CE] valid_tokens=152 ce_loss=8.234567 acc=0.0132
14:22:32.456789 [MODEL] path=joint fm_loss=0.123456 ce_loss=8.234567 acc=0.013... continuous_action=True
14:22:32.567890 [LOSS] step=0 loss=8.358023 fm_loss=0.123456 ce_loss=8.234567 acc=0.013...
```

## Add Logging To A New Model

1. Copy `_model_log` to the top of the model file.
2. Add `self._model_log_fh = None` in `__init__`.
3. Add `_model_log(self._model_log_fh, f"[TAG] ...")` in each important forward branch or checkpoint.
4. Inject `_model_log_fh` into the new model in the logging initialization block in `finetune.py`.

## Notes

- Do not replace this with `logger.info()`; it is unreliable inside compile regions.
- Flush every line so logs are not lost before a crash.
- Use one file per rank to avoid multi-process writes interleaving in one file.
- `.item()` triggers GPU synchronization. In production, log only the first N steps or every K steps if performance matters.
- Use append mode, `mode="a"`, so resume training does not overwrite previous logs.

---
Last modified: 2026-03-09
