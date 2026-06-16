from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from g05.models.g05.inferencer import PolicyInferencer


class FakeProcessor:
    pad_token_id = 0

    def preprocess(self, obs_dict: dict) -> dict:
        tokens = torch.tensor(obs_dict["tokens"], dtype=torch.long)
        proprio = obs_dict["proprio"].clone()
        action_horizon = obs_dict.get("action_horizon", 1)
        action_dim = proprio.shape[-1]
        return {
            "input_ids": tokens,
            "labels": torch.full_like(tokens, -100),
            "attention_mask": torch.ones(tokens.shape[0], dtype=torch.float32),
            "pixel_values": obs_dict["pixel_values"].clone(),
            "proprio": proprio,
            "proprio_is_pad": torch.zeros(proprio.shape[0], dtype=torch.bool),
            "proprio_dim_is_pad": torch.zeros(action_dim, dtype=torch.bool),
            "action": torch.zeros(action_horizon, action_dim, dtype=torch.float32),
            "action_is_pad": torch.ones(action_horizon, dtype=torch.bool),
            "action_dim_is_pad": torch.zeros(action_dim, dtype=torch.bool),
            "idx": obs_dict["idx"],
            "samples": {"idx": obs_dict["idx"]},
        }

    def postprocess(self, data: dict) -> dict:
        return {"action": {"arm": data["action"]}}


class FakePolicy:
    def predict_action(self, batch: dict) -> dict:
        seq_len = batch["attention_mask"].sum(dim=1, keepdim=True).unsqueeze(-1)
        batch["action"] = batch["proprio"][:, -1:, :] + seq_len
        return batch


def _make_obs(idx: int, tokens: list[int]) -> dict:
    return {
        "idx": idx,
        "tokens": tokens,
        "pixel_values": torch.full((1, 3, 4, 4), float(idx), dtype=torch.float32),
        "proprio": torch.tensor([[float(idx), float(idx + 1)]], dtype=torch.float32),
        "action_horizon": 1,
    }


@pytest.fixture(autouse=True)
def patch_cuda_path(monkeypatch):
    from g05.models.g05 import inferencer as inf_mod

    monkeypatch.setattr(inf_mod, "dict_apply", lambda x, func: x)
    monkeypatch.setattr(inf_mod.torch, "autocast", lambda *args, **kwargs: nullcontext())


def test_infer_matches_sequential_for_variable_lengths():
    policy = FakePolicy()
    processor = FakeProcessor()
    inferencer = PolicyInferencer(policy, processor)
    obs_dicts = [
        _make_obs(0, [1, 2, 3]),
        _make_obs(1, [4, 5, 6, 7, 8]),
        _make_obs(2, [9, 10, 11, 12, 13, 14, 15]),
    ]

    batched = inferencer.infer(obs_dicts)
    sequential = [inferencer.infer_one(obs_dict) for obs_dict in obs_dicts]

    assert len(batched) == len(sequential)
    for batch_action, single_action in zip(batched, sequential):
        assert torch.equal(batch_action["arm"], single_action["arm"])
