# G05 重构 -- Config 设计

> 更新于 2026.03.10
>
> **核心原则：谁拥有权重，谁负责初始化。VLM 和 Action Expert 各自完整独立一套 config，没有共享层级。**

## 1. Before / After

### Before -- `joint` 共享层 + OmegaConf.merge

```yaml
model_arch:
  vocab_size: 257216
  pad_token_id: 0
  joint:
    num_hidden_layers: 18       # 共享
    num_attention_heads: 8      # 共享
    num_key_value_heads: 1      # 共享
    head_dim: 256               # 共享
    rms_norm_eps: 0.000001      # 共享
    rope_theta: 10000.0         # 共享
    mixture:
      vlm:
        hidden_size: 2048       # 独有
        intermediate_size: 16384
      action:
        hidden_size: 1024       # 独有
        intermediate_size: 4096
        adaptive_mode: adaLN
```

```python
# JointModel.__init__ -- 隐式 merge
mixture_config = OmegaConf.merge(config, mixture_config)
```

### After -- 两套完整独立 config

```yaml
model_arch:
  pretrained_model_path: .../paligemma-3b-pt-224

  # VLM: 不写，全部从 HF config 自动派生
  # （也可以显式写来 override HF 值）

  # Action Expert: 完整独立一套
  action_expert:
    hidden_size: 1024
    intermediate_size: 4096
    num_hidden_layers: 18
    num_attention_heads: 8
    num_key_value_heads: 1
    head_dim: 256
    rms_norm_eps: 0.000001
    rope_theta: 10000.0
    attention_bias: false
    max_position_embeddings: 8192
    adaptive_mode: adaLN
    time_hidden_size: 1024
    input_type: linear
    input_dim: "${sum_shapes:${data.shape_meta.action}}"
    output_type: linear
    output_dim: "${sum_shapes:${data.shape_meta.action}}"
    use_final_norm: true
```

```python
# 没有 merge，没有 joint 层
vlm = Mixture.from_pretrained(pretrained_model_path)    # config + weights 从 HF
action_expert = Mixture(action_config)                   # config 从 YAML, weights 随机
```

## 2. Config 双模式

```
模式 1（推荐）：YAML 不写 VLM 参数 -> from_pretrained 从 HF 自动派生全部
模式 2（兼容）：YAML 显式写了某些参数 -> 覆盖 HF 值（用于微调架构参数）
```

实现：`OmegaConf.merge(hf_derived_config, yaml_overrides)`

## 3. Mixture 两个初始化入口

```python
class Mixture(nn.Module):
    def __init__(self, config):
        """随机初始化 -- Action Expert 用这个。
        config 必须包含完整一套参数。
        """

    @classmethod
    def from_pretrained(cls, pretrained_model_path=None, *,
                        hf_config=None, tensors=None, **overrides):
        """从 HF 预训练模型创建 VLM Mixture。
        1. 从 HF config 构建完整 config
        2. 用 overrides 覆盖（如果 YAML 显式指定了）
        3. 创建实例 + 加载权重
        """
        if hf_config is None:
            hf_config = AutoConfig.from_pretrained(pretrained_model_path)
        text_cfg = hf_config.text_config

        config = OmegaConf.create({
            "hidden_size": text_cfg.hidden_size,
            "intermediate_size": text_cfg.intermediate_size,
            "num_hidden_layers": text_cfg.num_hidden_layers,
            "num_attention_heads": text_cfg.num_attention_heads,
            "num_key_value_heads": text_cfg.num_key_value_heads,
            "head_dim": text_cfg.head_dim,
            "rms_norm_eps": text_cfg.rms_norm_eps,
            "rope_theta": text_cfg.rope_theta,
            "attention_bias": text_cfg.attention_bias,
            "max_position_embeddings": text_cfg.max_position_embeddings,
            "input_type": "embedding",
            "vocab_size": text_cfg.vocab_size,
            "pad_token_id": hf_config.pad_token_id,
            "output_type": "lm_head",
            "adaptive_mode": None,
            "use_final_norm": True,
        })

        if overrides:
            config = OmegaConf.merge(config, OmegaConf.create(overrides))

        model = cls(config)
        model._load_pretrained_weights(pretrained_model_path, tensors=tensors)
        return model

    def _load_pretrained_weights(self, pretrained_model_path, tensors=None):
        """加载 HF 权重，key 映射：
            language_model.model.embed_tokens.*  -> input_proj.*
            language_model.model.layers.*        -> layers.*
            language_model.model.norm.*          -> norm.*
        """
        if tensors is None:
            tensors = load_all_safetensors(pretrained_model_path)

        PREFIX_MAP = [
            ("language_model.model.embed_tokens.", "input_proj."),
            ("language_model.model.norm.",         "norm."),
            ("language_model.model.layers.",       "layers."),
        ]
        mapped = {}
        for key, tensor in tensors.items():
            for src, dst in PREFIX_MAP:
                if key.startswith(src):
                    mapped[key.replace(src, dst, 1)] = tensor
                    break

        # strict=False: output_proj (lm_head, tied) 不需要单独加载
        self.load_state_dict(mapped, strict=False)
```

## 4. G05Model 组装

```python
class G05Model(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_model_path, action_config, **kwargs):
        """首次训练：VLM 从 HF 加载，Action Expert 随机初始化。"""
        # 只加载一次 safetensors + HF config
        tensors = load_all_safetensors(pretrained_model_path)
        hf_config = AutoConfig.from_pretrained(pretrained_model_path)

        model = cls.__new__(cls)
        nn.Module.__init__(model)

        # 各组件独立 from_pretrained，共享 tensors
        model.vision_tower = SiglipVisionModel.from_pretrained(
            hf_config=hf_config, tensors=tensors)
        model.projector = PaliGemmaMultiModalProjector.from_pretrained(
            hf_config=hf_config, tensors=tensors)
        model.vlm = Mixture.from_pretrained(
            hf_config=hf_config, tensors=tensors)

        del tensors  # 释放 ~3.5GB

        model.action_expert = Mixture(action_config)  # 随机初始化
        return model

    def __init__(self, vlm_config, action_config, vision_config, **kwargs):
        """从 checkpoint 恢复：所有 config 从 checkpoint 读取。"""
        super().__init__()
        self.vlm = Mixture(vlm_config)
        self.action_expert = Mixture(action_config)
        self.vision_tower = SiglipVisionModel(vision_config)
        # ...
```

**首次训练 vs 恢复训练**：

```
首次训练：G05Model.from_pretrained(hf_path, action_cfg)
          +-- VLM:           config 从 HF, weights 从 HF
          +-- Vision:        config 从 HF, weights 从 HF
          +-- Projector:     config 从 HF, weights 从 HF
          +-- Action Expert: config 从 YAML, weights 随机

恢复训练：G05Model(saved_vlm_cfg, saved_action_cfg, ...)
          +-- load_state_dict(checkpoint)  -- 不走 HF from_pretrained
```

## 5. 换模型示例

### PaliGemma-3B -> Gemma-2-9B

```yaml
# 改 1 行，VLM 参数全部自动适配
pretrained_model_path: .../gemma-2-9b

# Vision 需要另外指定（Gemma-2 不含 vision）
vision:
  pretrained_model_path: .../siglip-so400m-patch14-224

# Action Expert 完整独立，人工调整
action_expert:
  hidden_size: 2048
  intermediate_size: 8192
  num_hidden_layers: 18
  num_attention_heads: 16
  num_key_value_heads: 8
  head_dim: 256
  # ...
```

非 PaliGemma 模型（无 vision_config）：

```python
if hasattr(hf_config, "vision_config"):
    model.vision_tower = SiglipVisionModel.from_pretrained(
        hf_config=hf_config, tensors=tensors)
else:
    model.vision_tower = SiglipVisionModel.from_pretrained(
        cfg.vision.pretrained_model_path)
```

## 6. Checkpoint 权重映射

旧 checkpoint -> 新 checkpoint 的 key 映射：

```python
WEIGHT_MAP = {
    # VLM embed_tokens -> vlm.input_proj
    "model.embed_tokens.weight":                      "model.vlm.input_proj.weight",
    # VLM lm_head -> vlm.output_proj (tied, 不需要单独映射)

    # VLM layers
    "model.joint_model.mixtures.vlm.layers.{i}.":    "model.vlm.layers.{i}.",
    "model.joint_model.mixtures.vlm.norm.":           "model.vlm.norm.",

    # Action Expert layers
    "model.joint_model.mixtures.action.layers.{i}.":  "model.action_expert.layers.{i}.",
    "model.joint_model.mixtures.action.norm.":         "model.action_expert.norm.",

    # Action encoder/decoder/time -> action_expert.*
    "model.action_encoder.":                           "model.action_expert.input_proj.",
    "model.action_decoder.":                           "model.action_expert.output_proj.",
    "model.time_embedding.":                           "model.action_expert.time_embedding.",
    "model.time_mlp_in.":                              "model.action_expert.time_mlp_in.",
    "model.time_mlp_out.":                             "model.action_expert.time_mlp_out.",

    # Vision (不变)
    "model.vision_tower.":                             "model.vision_tower.",
    "model.multi_modal_projector.":                    "model.multi_modal_projector.",
}
```

## 7. 风险和注意事项

1. **embed_tokens 扩展 vocab**：HF vocab < 我们的 vocab 时，`_load_pretrained_weights` 需保留 truncate 逻辑
2. **lm_head weight tying**：`output_proj.weight = input_proj.weight`，`strict=False` 自然跳过
3. **safetensors 只加载一次**：`G05Model.from_pretrained` 统一加载后传给各组件
4. **Action Expert n_kv_heads/head_dim 必须和 VLM 一致**（KV cache 兼容），在 YAML 里直接写

---

## 相关文档

- [G05 架构设计](g05_architecture.md) -- 层次结构、Mixture、Helper、精度策略
- [G05 I/O 格式](g05_io.md) -- 全链路 I/O 格式
