"""Tests for scheduler/instruction/instruction.py – InstructionManager."""

import numpy as np
import pytest
from unittest.mock import patch

from scheduler.instruction.instruction import InstructionManager, InstructionAction


def _make_config(**overrides):
    defaults = {
        "use_vlm": False,
        "image_as_condition": False,
        "bbox_as_instruction": False,
        "image_condition_lang_prefix": False,
        "pp_lower_half": False,
    }
    defaults.update(overrides)
    return defaults


def _make_obs():
    """Minimal obs dict that InstructionManager expects."""
    return {
        "images": {
            "head_rgb": np.zeros((3, 480, 640), dtype=np.uint8),
        },
    }


class TestGetInstruction:
    def test_empty_instruction_returns_skip(self):
        mgr = InstructionManager(_make_config())
        obs = _make_obs()
        with patch.object(mgr, "_get_instruction_from_file", return_value=""):
            action = mgr.get_instruction(obs)
        assert action == InstructionAction.SKIP
        assert obs["task"] == ""

    def test_nothing_returns_skip(self):
        mgr = InstructionManager(_make_config())
        obs = _make_obs()
        with patch.object(mgr, "_get_instruction_from_file", return_value="nothing"):
            action = mgr.get_instruction(obs)
        assert action == InstructionAction.SKIP
        assert obs["task"] == "nothing"

    def test_reset_returns_reset(self):
        mgr = InstructionManager(_make_config())
        obs = _make_obs()
        with patch.object(mgr, "_get_instruction_from_file", return_value="reset"):
            action = mgr.get_instruction(obs)
        assert action == InstructionAction.RESET
        assert obs["task"] == "reset"

    def test_new_instruction_returns_continue(self):
        mgr = InstructionManager(_make_config())
        obs = _make_obs()
        with patch.object(mgr, "_get_instruction_from_file", return_value="pick up the cup"):
            action = mgr.get_instruction(obs)
        assert action == InstructionAction.CONTINUE
        assert obs["task"] == "pick up the cup"

    def test_repeated_instruction_reuses_extra_info(self):
        mgr = InstructionManager(_make_config())
        obs1 = _make_obs()
        with patch.object(mgr, "_get_instruction_from_file", return_value="pick up the cup"):
            mgr.get_instruction(obs1)

        # Same instruction again → should still return CONTINUE and reuse extra_info
        obs2 = _make_obs()
        with patch.object(mgr, "_get_instruction_from_file", return_value="pick up the cup"):
            action = mgr.get_instruction(obs2)
        assert action == InstructionAction.CONTINUE
        assert obs2["task"] == "pick up the cup"

    def test_instruction_change_updates_task(self):
        mgr = InstructionManager(_make_config())
        obs1 = _make_obs()
        with patch.object(mgr, "_get_instruction_from_file", return_value="pick up the cup"):
            mgr.get_instruction(obs1)

        obs2 = _make_obs()
        with patch.object(mgr, "_get_instruction_from_file", return_value="put down the cup"):
            action = mgr.get_instruction(obs2)
        assert action == InstructionAction.CONTINUE
        assert obs2["task"] == "put down the cup"


class TestRefineInstruction:
    def test_already_has_low_tag(self):
        mgr = InstructionManager(_make_config())
        assert mgr._refine_ll_instruction("[Low] pick cup") == "[Low] pick cup"

    def test_lowercase_low_tag(self):
        mgr = InstructionManager(_make_config())
        assert mgr._refine_ll_instruction("[low] pick cup") == "[Low] pick cup"

    def test_no_tag_adds_low(self):
        mgr = InstructionManager(_make_config())
        assert mgr._refine_ll_instruction("pick cup") == "[Low]:pick cup"

    def test_reset_passthrough(self):
        mgr = InstructionManager(_make_config())
        assert mgr._refine_ll_instruction("reset") == "reset"


class TestConfigValidation:
    def test_image_condition_requires_vlm(self):
        with pytest.raises(ValueError, match="require use_vlm=true"):
            InstructionManager(_make_config(image_as_condition=True))

    def test_bbox_instruction_requires_vlm(self):
        with pytest.raises(ValueError, match="require use_vlm=true"):
            InstructionManager(_make_config(bbox_as_instruction=True))
