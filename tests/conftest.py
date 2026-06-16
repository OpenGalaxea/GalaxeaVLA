"""
pytest fixtures for G0.5 model tests.

Level 1 (inner model): CPU, no pretrained weights
Level 2 (policy):      GPU + pretrained weights
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import pytest
import torch
import numpy as np

# Ensure project root is on sys.path so `scripts/` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Register OmegaConf resolvers (same set as finetune.py)
# ---------------------------------------------------------------------------
from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()

from omegaconf import OmegaConf

# Register Hydra built-in resolvers that aren't available outside Hydra
OmegaConf.register_new_resolver(
    "now", lambda pattern, _tz="": datetime.now().strftime(pattern), replace=True
)
OmegaConf.register_new_resolver(
    "oc.env",
    lambda key, default=None: (
        os.environ.get(key, default) if default is not None else os.environ[key]
    ),
    replace=True,
)


# ---------------------------------------------------------------------------
# pytest CLI options
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--task-config",
        default=None,
        help="Hydra task config name, e.g. 'test/model_unit_test'",
    )
    parser.addoption(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra overrides, repeatable (e.g. --override model.model_arch.action_dim=32)",
    )


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "pretrained: requires pretrained model weights")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def resolved_cfg(request):
    """Load task config via Hydra compose — same resolution as finetune.sh.

    Priority: --task-config CLI flag > PYTEST_TASK_CONFIG env var > default 'libero'.
    Extra Hydra overrides can be passed via --override (repeatable).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    task_cfg = (
        request.config.getoption("--task-config", default=None)
        or os.environ.get("PYTEST_TASK_CONFIG")
        or "libero"
    )
    extra_overrides = request.config.getoption("--override", default=[])

    config_dir = str(PROJECT_ROOT / "configs")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"task={task_cfg}"] + list(extra_overrides),
        )
    return cfg


@pytest.fixture(scope="session")
def model_arch_cfg(resolved_cfg):
    """Return the resolved model_arch DictConfig (used by GalaxeaJoint)."""
    return resolved_cfg.model.model_arch


# ---------------------------------------------------------------------------
# Synthetic batch helpers
# ---------------------------------------------------------------------------
def make_synthetic_inner_batch(model_arch_cfg, device, B=2):
    """Create a synthetic tensor batch for GalaxeaJoint.forward() (Level 1).

    All values are from the resolved config so dimensions match the real model.
    """
    cfg = model_arch_cfg
    H = OmegaConf.to_container(cfg, resolve=True).get("horizon_steps", 32)
    A = OmegaConf.to_container(cfg, resolve=True).get("action_dim", 79)
    vocab_size = cfg.vocab_size
    max_imt = cfg.max_image_text_tokens
    num_imgs = cfg.num_input_images

    # Total sequence length = prefix (max_image_text_tokens) + suffix action tokens
    suffix_len = 50  # rough number of action tokens in suffix
    seq_len = max_imt + suffix_len

    # attention_mask: IMAGE=1 for image tokens, TEXT=4 for text, ACTION=3 for suffix
    # Use TOKEN_INDEX values
    IMAGE_TOKEN_INDEX = 1
    TEXT_TOKEN_INDEX = 4
    ACTION_TOKEN_INDEX = 3

    num_image_tokens = num_imgs * 256  # 256 patch tokens per image
    num_text_tokens = max_imt - num_image_tokens

    attn_mask = torch.zeros(B, seq_len, dtype=torch.long, device=device)
    attn_mask[:, :num_image_tokens] = IMAGE_TOKEN_INDEX
    attn_mask[:, num_image_tokens:max_imt] = TEXT_TOKEN_INDEX
    attn_mask[:, max_imt:] = ACTION_TOKEN_INDEX

    # input_ids: random valid token ids. Use image_token_index (257152) for image positions
    input_ids = torch.randint(1, 1000, (B, seq_len), device=device)
    input_ids[:, :num_image_tokens] = cfg.image_token_index  # 257152

    # labels: -100 for prefix (no CE loss), random action token ids for suffix
    labels = torch.full((B, seq_len), -100, dtype=torch.long, device=device)
    labels[:, max_imt:] = torch.randint(1, 1000, (B, suffix_len), device=device)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
        pixel_values=torch.randn(B, num_imgs, 3, 224, 224, device=device),
        actions=torch.randn(B, H, A, device=device),
        action_pad_masks=torch.zeros(B, H, dtype=torch.bool, device=device),
        action_dim_is_pad=torch.zeros(B, A, dtype=torch.bool, device=device),
        split_index=max_imt,
        t=torch.rand(B, device=device),
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Level 2 shared helpers: batch helpers
# ---------------------------------------------------------------------------
def make_policy_batch(model_arch_cfg, device, B=2):
    """Create synthetic batch for GalaxeaJointPolicy.forward().

    Structure mirrors real batches from the training pipeline (verified against
    forward_train_debug.pth).
    """
    cfg = OmegaConf.to_container(model_arch_cfg, resolve=True)
    H = cfg["horizon_steps"]
    A = cfg["action_dim"]
    num_imgs = cfg["num_input_images"]

    # Build image placeholder keys: image0, image1, ... (value=0, just an index)
    image_placeholders = "".join(f"<image{i}_image_!>" for i in range(num_imgs))
    template = (
        f"{image_placeholders}<bos>Task: <command_text_!_200> "
        f"State: <proprio_proprio_!>;\n<EOV><EOC>Action: <action_action>|<eos>"
    )

    samples = []
    for _ in range(B):
        sample = {
            "template": template,
            "command": "pick up the red block",
            "proprio": {
                "value": torch.randn(1, A),  # (obs_size, proprio_dim)
                "proprio_dim_is_pad": torch.zeros(A, dtype=torch.bool),
                "embodiment": "galaxea_r1lite",
            },
            "action": {
                "value": torch.randn(H, A),
                "action_dim_is_pad": torch.zeros(A, dtype=torch.bool),
                "action_op_mask": torch.ones(1, A, dtype=torch.bool),
                "embodiment": "galaxea_r1lite",
            },
        }
        # image keys: image0=0, image1=0, ...
        for i in range(num_imgs):
            sample[f"image{i}"] = 0
        samples.append(sample)

    batch = {
        "samples": samples,
        "pixel_values": torch.randn(B, num_imgs, 3, 224, 224, device=device),
        "action": torch.randn(B, H, A, device=device),
        "action_is_pad": torch.zeros(B, H, dtype=torch.bool, device=device),
        "action_dim_is_pad": torch.zeros(B, A, dtype=torch.bool, device=device),
    }
    return batch


# ---------------------------------------------------------------------------
# Synthetic dataset stats generation
# ---------------------------------------------------------------------------
def make_synthetic_stats(shape_meta, seed=42):
    """Generate synthetic dataset stats matching a given shape_meta.

    Returns stats dict compatible with LinearNormalizer:
        {
            "action": { key: {"global_mean": ..., "global_std": ..., ...} },
            "state":  { key: {"global_mean": ..., "global_std": ..., ...} },
        }
    """
    rng = np.random.RandomState(seed)
    stats = {"action": {}, "state": {}}

    for category in ("action", "state"):
        for meta in shape_meta[category]:
            key = meta["key"]
            dim = meta["shape"]
            # Generate reasonable synthetic statistics
            low = rng.uniform(-2.0, -0.5, size=dim).astype(np.float32)
            high = low + rng.uniform(0.5, 3.0, size=dim).astype(np.float32)
            mean = ((low + high) / 2).astype(np.float32)
            std = ((high - low) / 4).astype(np.float32)  # ~4 sigma coverage

            field_stats = {
                "global_min": torch.from_numpy(low),
                "global_max": torch.from_numpy(high),
                "global_mean": torch.from_numpy(mean),
                "global_std": torch.from_numpy(std),
                "global_q01": torch.from_numpy(low + (high - low) * 0.01),
                "global_q99": torch.from_numpy(low + (high - low) * 0.99),
                "global_q001": torch.from_numpy(low + (high - low) * 0.001),
                "global_q999": torch.from_numpy(low + (high - low) * 0.999),
                "global_q0001": torch.from_numpy(low + (high - low) * 0.0001),
                "global_q9999": torch.from_numpy(low + (high - low) * 0.9999),
                "global_q00001": torch.from_numpy(low + (high - low) * 0.00001),
                "global_q99999": torch.from_numpy(low + (high - low) * 0.99999),
                # stepwise stats (same as global for synthetic)
                "stepwise_min": torch.from_numpy(low),
                "stepwise_max": torch.from_numpy(high),
                "stepwise_mean": torch.from_numpy(mean),
                "stepwise_std": torch.from_numpy(std),
                "stepwise_q01": torch.from_numpy(low + (high - low) * 0.01),
                "stepwise_q99": torch.from_numpy(low + (high - low) * 0.99),
                "stepwise_q001": torch.from_numpy(low + (high - low) * 0.001),
                "stepwise_q999": torch.from_numpy(low + (high - low) * 0.999),
                "stepwise_q0001": torch.from_numpy(low + (high - low) * 0.0001),
                "stepwise_q9999": torch.from_numpy(low + (high - low) * 0.9999),
                "stepwise_q00001": torch.from_numpy(low + (high - low) * 0.00001),
                "stepwise_q99999": torch.from_numpy(low + (high - low) * 0.99999),
            }
            stats[category][key] = field_stats

    return stats


def make_dict_action_state(shape_meta, H=20, num_obs_steps=1, seed=0):
    """Create synthetic dict-format action & state data matching shape_meta.

    Returns a batch dict with:
        - action: Dict[str, Tensor] shaped (H, dim) for each key
        - state:  Dict[str, Tensor] shaped (num_obs_steps, dim) for each key
    Values are drawn from a moderate range to stay within normalizer bounds.
    """
    rng = np.random.RandomState(seed)
    action = {}
    state = {}

    for meta in shape_meta["action"]:
        key, dim = meta["key"], meta["shape"]
        action[key] = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(H, dim)).astype(np.float32))

    for meta in shape_meta["state"]:
        key, dim = meta["key"], meta["shape"]
        state[key] = torch.from_numpy(
            rng.uniform(-1.0, 1.0, size=(num_obs_steps, dim)).astype(np.float32)
        )

    return {"action": action, "state": state}
