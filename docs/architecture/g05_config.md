# G05 Refactor: Config Design

> Updated on 2026-03-10
>
> Core principle: the component that owns weights is responsible for initialization. VLM and Action Expert each have a complete independent config, with no shared `joint` layer.

## 1. Before / After

### Before: Shared `joint` Layer + `OmegaConf.merge`

```yaml
model_arch:
  vocab_size: 257216
  pad_token_id: 0
  joint:
    num_hidden_layers: 18       # shared
    num_attention_heads: 8      # shared
    num_key_value_heads: 1      # shared
    head_dim: 256               # shared
    rms_norm_eps: 0.000001      # shared
    rope_theta: 10000.0         # shared
    mixture:
      vlm:
        hidden_size: 2048       # VLM-only
        intermediate_size: 16384
      action:
        hidden_size: 1024       # action-only
        intermediate_size: 4096
        adaptive_mode: adaLN
```

```python
# JointModel.__init__: implicit merge
mixture_config = OmegaConf.merge(config, mixture_config)
```

### After: Two Complete Independent Configs

```yaml
model_arch:
  pretrained_model_path: .../paligemma-3b-pt-224

  # VLM: omit fields to derive everything automatically from HF config.
  # Explicit YAML values can still override HF values when needed.

  # Action Expert: complete standalone config.
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
# No merge, no joint layer.
vlm = Mixture.from_pretrained(pretrained_model_path)    # config + weights from HF
action_expert = Mixture(action_config)                  # config from YAML, random weights
```

## 2. Dual Config Modes

```text
Mode 1, recommended: YAML omits VLM parameters -> from_pretrained derives all fields from HF.
Mode 2, compatibility: YAML explicitly defines some VLM parameters -> override HF values for architecture tuning.
```

Implementation: `OmegaConf.merge(hf_derived_config, yaml_overrides)`.

## 3. Two Mixture Initialization Entries

```python
class Mixture(nn.Module):
    def __init__(self, config):
        """Random initialization, used by Action Expert.
        The config must contain the full parameter set.
        """

    @classmethod
    def from_pretrained(cls, pretrained_model_path=None, *,
                        hf_config=None, tensors=None, **overrides):
        """Create the VLM Mixture from an HF pretrained model.
        1. Build a full config from the HF config.
        2. Apply overrides when YAML explicitly provides them.
        3. Create the module and load weights.
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
        """Load HF weights with key mapping:
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

        # strict=False: output_proj is tied to lm_head and does not need separate loading.
        self.load_state_dict(mapped, strict=False)
```

## 4. G05Model Assembly

```python
class G05Model(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_model_path, action_config, **kwargs):
        """First training run: load VLM from HF and randomly initialize Action Expert."""
        # Load safetensors + HF config once.
        tensors = load_all_safetensors(pretrained_model_path)
        hf_config = AutoConfig.from_pretrained(pretrained_model_path)

        model = cls.__new__(cls)
        nn.Module.__init__(model)

        # Each component owns its own from_pretrained path while sharing tensors.
        model.vision_tower = SiglipVisionModel.from_pretrained(
            hf_config=hf_config, tensors=tensors)
        model.projector = PaliGemmaMultiModalProjector.from_pretrained(
            hf_config=hf_config, tensors=tensors)
        model.vlm = Mixture.from_pretrained(
            hf_config=hf_config, tensors=tensors)

        del tensors  # releases about 3.5GB

        model.action_expert = Mixture(action_config)  # random initialization
        return model

    def __init__(self, vlm_config, action_config, vision_config, **kwargs):
        """Checkpoint resume: all configs are read from the checkpoint."""
        super().__init__()
        self.vlm = Mixture(vlm_config)
        self.action_expert = Mixture(action_config)
        self.vision_tower = SiglipVisionModel(vision_config)
        # ...
```

First training vs resume:

```text
First training: G05Model.from_pretrained(hf_path, action_cfg)
          +-- VLM:           config from HF, weights from HF
          +-- Vision:        config from HF, weights from HF
          +-- Projector:     config from HF, weights from HF
          +-- Action Expert: config from YAML, random weights

Resume: G05Model(saved_vlm_cfg, saved_action_cfg, ...)
          +-- load_state_dict(checkpoint), without HF from_pretrained
```

## 5. Example: Swapping Models

### PaliGemma-3B To Gemma-2-9B

```yaml
# Change one line; VLM parameters adapt automatically.
pretrained_model_path: .../gemma-2-9b

# Vision must be specified separately because Gemma-2 has no vision component.
vision:
  pretrained_model_path: .../siglip-so400m-patch14-224

# Action Expert is fully independent and manually adjusted.
action_expert:
  hidden_size: 2048
  intermediate_size: 8192
  num_hidden_layers: 18
  num_attention_heads: 16
  num_key_value_heads: 8
  head_dim: 256
  # ...
```

For non-PaliGemma models without `vision_config`:

```python
if hasattr(hf_config, "vision_config"):
    model.vision_tower = SiglipVisionModel.from_pretrained(
        hf_config=hf_config, tensors=tensors)
else:
    model.vision_tower = SiglipVisionModel.from_pretrained(
        cfg.vision.pretrained_model_path)
```

## 6. Checkpoint Weight Mapping

Old checkpoint keys map to new checkpoint keys as follows:

```python
WEIGHT_MAP = {
    # VLM embed_tokens -> vlm.input_proj
    "model.embed_tokens.weight":                      "model.vlm.input_proj.weight",
    # VLM lm_head -> vlm.output_proj, tied and not mapped separately

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

    # Vision, unchanged
    "model.vision_tower.":                             "model.vision_tower.",
    "model.multi_modal_projector.":                    "model.multi_modal_projector.",
}
```

## 7. Risks And Notes

1. **Extended embed_tokens vocabulary**: when the HF vocab is smaller than the project vocab, `_load_pretrained_weights` must keep the truncation logic.
2. **lm_head weight tying**: `output_proj.weight = input_proj.weight`; `strict=False` naturally skips it.
3. **safetensors loaded once**: `G05Model.from_pretrained` loads tensors once and passes them to each component.
4. **Action Expert `n_kv_heads` and `head_dim` must match VLM** for KV-cache compatibility; write them explicitly in YAML.

## Related Documents

- [G05 Architecture Design](g05_architecture.md): hierarchy, Mixture, Helper, and precision policy.
- [G05 I/O Format](g05_io.md): end-to-end I/O format.
