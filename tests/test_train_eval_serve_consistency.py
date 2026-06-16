"""
Three-way consistency check: Training batch vs eval_open_loop vs serve_policy.

Load a real training batch (pkl), compare eval_open_loop preprocess output with
serve_policy preprocess output, and detect train/inference inconsistency.

Usage:
    python tests/test_train_eval_serve_consistency.py \
        --ckpt_path runs/pretrain/.../checkpoints/step_117428.pt \
        --train_batch path/to/real_batch.pkl \
        eval_embodiment=galaxea_r1lite

Checks:
    A. Template: whether template structure and placeholders match; most critical
    B. Tensor shapes: input_ids, pixel_values, proprio, etc.
    C. dim_pad_mask: action_dim_is_pad / proprio_dim_is_pad dimensions
    D. Embodiment: whether the embodiment field is injected correctly
    E. Eval vs Serve value comparison: whether two paths match on the same input
    F. Structural completeness: whether samples subfields are complete
"""

from __future__ import annotations

import argparse
import logging
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()

from hydra.utils import instantiate
from omegaconf import OmegaConf

from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset
from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.utils.data.processor_utils import build_processors
from g05.utils.eval.eval_utils import filter_embodiment, truncate_datasets
from g05.utils.data.normalizer import load_dataset_stats_from_json

from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from scripts.serve_policy import build_obs_dict

logger = logging.getLogger(__name__)

# Keys produced only by collate; it is normal for preprocess not to include them.
COLLATE_ONLY_KEYS = {"embodiment", "frequency", "gt_action"}

# Key alias mapping inside samples.action/proprio.
# Training batch pkl stores post-collate names, while preprocess emits aliases.
ALIAS_MAP = {
    "action_dim_is_pad": "pad_mask",
    "action_op_mask": "op_mask",
    "proprio_dim_is_pad": "pad_mask",
}


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────


class ConsistencyReport:
    """Accumulate check results with PASS/FAIL/WARN/INFO."""

    # ANSI color codes
    _GREEN = "\033[32m"
    _RED = "\033[31m"
    _YELLOW = "\033[33m"
    _BLUE = "\033[34m"
    _CYAN = "\033[36m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"
    _BG_RED = "\033[41m"
    _BG_GREEN = "\033[42m"
    _WHITE = "\033[97m"

    def __init__(self):
        self.results: list[tuple[str, str, str]] = []

    def ok(self, name: str, detail: str = ""):
        self.results.append(("PASS", name, detail))

    def fail(self, name: str, detail: str):
        self.results.append(("FAIL", name, detail))

    def warn(self, name: str, detail: str):
        self.results.append(("WARN", name, detail))

    def info(self, name: str, detail: str):
        self.results.append(("INFO", name, detail))

    def print_report(self):
        G, R, Y, B, C = self._GREEN, self._RED, self._YELLOW, self._BLUE, self._CYAN
        BOLD, DIM, RST = self._BOLD, self._DIM, self._RESET
        BG_R, BG_G, W = self._BG_RED, self._BG_GREEN, self._WHITE

        n_pass = sum(1 for s, _, _ in self.results if s == "PASS")
        n_fail = sum(1 for s, _, _ in self.results if s == "FAIL")
        n_warn = sum(1 for s, _, _ in self.results if s == "WARN")
        n_info = sum(1 for s, _, _ in self.results if s == "INFO")

        print(f"\n{BOLD}{C}{'━' * 80}{RST}")
        print(
            f"{BOLD}  📋 CONSISTENCY REPORT  "
            f"{G}✅ {n_pass} PASS{RST}  "
            f"{R}❌ {n_fail} FAIL{RST}  "
            f"{Y}⚠️  {n_warn} WARN{RST}  "
            f"{B}ℹ️  {n_info} INFO{RST}"
        )
        print(f"{BOLD}{C}{'━' * 80}{RST}\n")

        STATUS_CFG = {
            "PASS": (f"{G}✅", G, ""),
            "FAIL": (f"{R}❌", R, f"{R}▸▸▸ "),
            "WARN": (f"{Y}⚠️ ", Y, f"{Y} ▸  "),
            "INFO": (f"{B}ℹ️ ", DIM, f"{DIM}     "),
        }

        for status, name, detail in self.results:
            icon, color, prefix = STATUS_CFG[status]
            line = f"{prefix}  {icon} {color}{name}{RST}"
            if detail:
                detail_color = R if status == "FAIL" else (Y if status == "WARN" else DIM)
                line += f"\n        {detail_color}{detail}{RST}"
            print(line)

        print(f"\n{BOLD}{C}{'━' * 80}{RST}")
        if n_fail > 0:
            print(
                f"  {BG_R}{W}{BOLD} 💥 {n_fail} FAILURES detected — training-inference inconsistency! {RST}"
            )
        else:
            print(f"  {BG_G}{W}{BOLD} 🎉 All critical checks passed. {RST}")
        print(f"{BOLD}{C}{'━' * 80}{RST}\n")

    def to_text(self) -> str:
        lines = []
        for status, name, detail in self.results:
            lines.append(f"[{status}] {name}")
            if detail:
                lines.append(f"       {detail}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _extract_template_placeholders(template: str) -> list[str]:
    """Extract placeholder names like 'embodiment' from <embodiment_text_!>."""
    return re.findall(r"<(\w+?)_\w+?_[!][\w]*>", template)


def _get_samples(preprocessed: dict) -> dict:
    """Extract 'samples' sub-dict from preprocess output."""
    s = preprocessed.get("samples", {})
    if isinstance(s, list) and s:
        return s[0]
    return s if isinstance(s, dict) else {}


def _get_train_emb(train_batch: dict) -> str | None:
    """Detect embodiment from training batch."""
    if "embodiment" in train_batch:
        emb_list = train_batch["embodiment"]
        if isinstance(emb_list, list) and emb_list:
            return emb_list[0]
    if "samples" in train_batch:
        return train_batch["samples"][0].get("embodiment")
    return None


# ──────────────────────────────────────────────────────────────────────
# Check A: Template (most critical; directly affects input_ids)
# ──────────────────────────────────────────────────────────────────────


def check_template(
    report: ConsistencyReport, train_template: str, other_template: str, other_name: str
):
    """Compare template structure — mismatch means input_ids differ."""
    train_ph = _extract_template_placeholders(train_template)
    other_ph = _extract_template_placeholders(other_template)

    if train_ph == other_ph:
        report.ok(
            f"A. Template placeholders: {other_name} matches training", f"Placeholders: {train_ph}"
        )
    else:
        report.fail(
            f"A. Template placeholders: {other_name} differs from training",
            f"Training:  {train_ph}\n      {other_name}: {other_ph}",
        )

    if train_template == other_template:
        report.ok(f"A. Template text: {other_name} exact match")
    else:
        report.fail(
            f"A. Template text: {other_name} differs",
            f"Training:  {repr(train_template[:120])}\n"
            f"      {other_name}: {repr(other_template[:120])}",
        )


# ──────────────────────────────────────────────────────────────────────
# Check B: Tensor shapes
# ──────────────────────────────────────────────────────────────────────

CRITICAL_FIELDS = [
    "input_ids",
    "attention_mask",
    "labels",
    "pixel_values",
    "proprio",
    "action_dim_is_pad",
    "proprio_dim_is_pad",
]


def check_tensor_shapes(
    report: ConsistencyReport, train_batch: dict, other_sample: dict, other_name: str
):
    for field in CRITICAL_FIELDS:
        if field not in train_batch or field not in other_sample:
            if field in train_batch and field not in other_sample:
                report.fail(f"B. Shape {field}: missing in {other_name}", "")
            continue

        t_val, o_val = train_batch[field], other_sample[field]
        if not isinstance(t_val, torch.Tensor) or not isinstance(o_val, torch.Tensor):
            continue

        t_shape = list(t_val.shape[1:])  # remove batch dim
        o_shape = list(o_val.shape)

        if t_shape == o_shape:
            report.ok(f"B. Shape {field}: {other_name} [{o_shape}]")
        else:
            report.fail(
                f"B. Shape {field}: mismatch", f"Training: {t_shape}, {other_name}: {o_shape}"
            )

        if t_val.dtype != o_val.dtype:
            report.warn(f"B. Dtype {field}: train={t_val.dtype}, {other_name}={o_val.dtype}", "")


# ──────────────────────────────────────────────────────────────────────
# Check C: dim_pad_mask (padding counts may differ across embodiments by design)
# ──────────────────────────────────────────────────────────────────────


def check_dim_pad_masks(
    report: ConsistencyReport,
    train_batch: dict,
    other_sample: dict,
    other_name: str,
    same_embodiment: bool,
):
    for field in ["action_dim_is_pad", "proprio_dim_is_pad"]:
        if field not in train_batch or field not in other_sample:
            continue
        train_mask = train_batch[field][0] if train_batch[field].ndim > 1 else train_batch[field]
        other_mask = other_sample[field]

        train_n = train_mask.sum().item()
        other_n = other_mask.sum().item()
        total = train_mask.shape[0]

        if torch.equal(train_mask, other_mask):
            report.ok(f"C. {field}: {other_name} matches ({train_n}/{total} padded)")
        elif same_embodiment:
            report.fail(
                f"C. {field}: {other_name} differs (SAME embodiment!)",
                f"Training: {train_n}/{total}, {other_name}: {other_n}/{total}",
            )
        else:
            report.info(
                f"C. {field}: differs (expected — different embodiment)",
                f"Training: {train_n}/{total}, {other_name}: {other_n}/{total}",
            )

    # Total dim must match; merger output dim is 79 for all embodiments.
    if "action_dim_is_pad" in train_batch and "action_dim_is_pad" in other_sample:
        t_dim = train_batch["action_dim_is_pad"].shape[-1]
        o_dim = other_sample["action_dim_is_pad"].shape[-1]
        if t_dim == o_dim:
            report.ok(f"C. Merged action dim: {other_name} matches ({o_dim})")
        else:
            report.fail(
                f"C. Merged action dim: mismatch", f"Training: {t_dim}, {other_name}: {o_dim}"
            )


# ──────────────────────────────────────────────────────────────────────
# Check D: Embodiment field presence
# ──────────────────────────────────────────────────────────────────────


def check_embodiment_fields(
    report: ConsistencyReport, train_batch: dict, other_sample: dict, other_name: str
):
    # Top-level 'embodiment' is a collate artifact; it is normal for preprocess to omit it.
    if "embodiment" in train_batch and "embodiment" not in other_sample:
        report.info(f"D. Top-level 'embodiment': absent in {other_name} (collate-only field)", "")

    # samples.embodiment is produced by preprocess for template <embodiment_text_!>.
    train_samples = train_batch.get("samples", [{}])
    ts = train_samples[0] if isinstance(train_samples, list) else train_samples
    other_s = _get_samples(other_sample)

    if "embodiment" in ts:
        if "embodiment" in other_s:
            report.ok(f"D. samples.embodiment: present in {other_name}")
        else:
            report.fail(
                f"D. samples.embodiment: MISSING in {other_name}",
                f"Training has '{ts['embodiment']}' — needed for <embodiment_text_!> placeholder",
            )


# ──────────────────────────────────────────────────────────────────────
# Check E: Eval vs Serve value consistency on the same input
# ──────────────────────────────────────────────────────────────────────


def check_values_eval_vs_serve(report: ConsistencyReport, eval_s: dict, serve_s: dict):
    for field in CRITICAL_FIELDS:
        if field not in eval_s or field not in serve_s:
            continue
        e_val, s_val = eval_s[field], serve_s[field]
        if not isinstance(e_val, torch.Tensor) or not isinstance(s_val, torch.Tensor):
            continue
        if e_val.shape != s_val.shape:
            report.fail(
                f"E. eval==serve {field}: shape mismatch",
                f"eval={list(e_val.shape)}, serve={list(s_val.shape)}",
            )
            continue

        if e_val.is_floating_point():
            max_diff = (e_val.float() - s_val.float()).abs().max().item()
            if max_diff < 1e-6:
                report.ok(f"E. eval==serve {field}: exact match")
            elif max_diff < 1e-3:
                report.ok(f"E. eval==serve {field}: close (max_diff={max_diff:.2e})")
            else:
                report.fail(f"E. eval==serve {field}: diverged", f"max_diff={max_diff:.6e}")
        else:
            if torch.equal(e_val, s_val):
                report.ok(f"E. eval==serve {field}: exact match")
            else:
                n_diff = (e_val != s_val).sum().item()
                report.fail(f"E. eval==serve {field}: {n_diff}/{e_val.numel()} differ", "")


# ──────────────────────────────────────────────────────────────────────
# Check F: Structure — samples subfield completeness
# ──────────────────────────────────────────────────────────────────────


def check_structure(
    report: ConsistencyReport, train_batch: dict, other_sample: dict, other_name: str
):
    """Check structural completeness, accounting for collate-only keys and aliases."""
    # F1: top-level keys, excluding collate-only keys.
    train_keys = {k for k in train_batch if k != "samples"} - COLLATE_ONLY_KEYS
    other_keys = set(other_sample.keys()) - {"samples"}
    missing = train_keys - other_keys
    if not missing:
        report.ok(f"F. Top-level keys: {other_name} complete")
    else:
        report.fail(f"F. Top-level keys: {other_name} missing", f"Missing: {sorted(missing)}")

    # F2: samples keys
    ts = train_batch["samples"][0] if "samples" in train_batch else {}
    os = _get_samples(other_sample)
    if ts and os:
        ts_keys = set(ts.keys())
        os_keys = set(os.keys())
        missing = ts_keys - os_keys
        if not missing:
            report.ok(f"F. samples keys: {other_name} complete")
        else:
            report.warn(f"F. samples keys: {other_name} missing {sorted(missing)}", "")

    # F3: samples.action sub-keys, alias-aware.
    if ts and isinstance(ts.get("action"), dict) and isinstance(os.get("action"), dict):
        t_action_keys = set(ts["action"].keys())
        o_action_keys = set(os["action"].keys())
        # Convert training keys to preprocess aliases.
        t_aliased = {ALIAS_MAP.get(k, k) for k in t_action_keys} - {"embodiment"}
        o_clean = o_action_keys
        missing = t_aliased - o_clean
        if not missing:
            report.ok(f"F. samples.action keys: {other_name} complete (alias-aware)")
        else:
            report.fail(
                f"F. samples.action keys: {other_name} missing",
                f"Expected (aliased): {sorted(t_aliased)}, Got: {sorted(o_clean)}",
            )


# ──────────────────────────────────────────────────────────────────────
# Check G: input_ids token-level comparison (training vs eval/serve)
# ──────────────────────────────────────────────────────────────────────


def check_input_ids_structure(
    report: ConsistencyReport, train_batch: dict, other_sample: dict, other_name: str
):
    """Compare input_ids structure — special token positions, padding pattern."""
    if "input_ids" not in train_batch or "input_ids" not in other_sample:
        return

    train_ids = train_batch["input_ids"][0]  # [seq_len]
    other_ids = other_sample["input_ids"]  # [seq_len]

    if train_ids.shape != other_ids.shape:
        report.fail(
            f"G. input_ids length: {other_name} mismatch",
            f"Training: {train_ids.shape[0]}, {other_name}: {other_ids.shape[0]}",
        )
        return

    # Count how many token positions differ.
    diff_mask = train_ids != other_ids
    n_diff = diff_mask.sum().item()
    total = train_ids.shape[0]

    if n_diff == 0:
        report.ok(f"G. input_ids: {other_name} exact match with training ({total} tokens)")
    else:
        # Find the first differing position.
        first_diff = diff_mask.nonzero(as_tuple=True)[0][0].item()
        report.warn(
            f"G. input_ids: {other_name} {n_diff}/{total} tokens differ from training",
            f"First diff at position {first_diff} "
            f"(train={train_ids[first_diff].item()}, {other_name}={other_ids[first_diff].item()}). "
            f"Note: different embodiment / task text cause expected differences.",
        )

    # Check padding token pattern; trailing pad regions should match.
    train_attn = train_batch.get("attention_mask")
    other_attn = other_sample.get("attention_mask")
    if train_attn is not None and other_attn is not None:
        train_real = train_attn[0].sum().item()
        other_real = other_attn.sum().item()
        report.info(
            f"G. Effective tokens: training={train_real}/{total}, {other_name}={other_real}/{total}",
            "",
        )


# ──────────────────────────────────────────────────────────────────────
# dataset sample -> simulated raw_obs
# ──────────────────────────────────────────────────────────────────────


def dataset_sample_to_raw_obs(sample: dict, embodiment_type: str | None = None) -> dict:
    """Convert dataset __getitem__ output to client raw_obs format."""
    raw_obs = {
        "images": {},
        "state": {},
        "task": sample["task"],
    }
    for k, v in sample["images"].items():
        raw_obs["images"][k] = v[-1].numpy()
    for k, v in sample["state"].items():
        raw_obs["state"][k] = v[-1].numpy()
    if embodiment_type:
        raw_obs["embodiment_type"] = embodiment_type
    return raw_obs


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Three-way consistency: training batch vs eval vs serve"
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--train_batch", required=True, help="Path to real_batch.pkl from training")
    parser.add_argument("--max_datasets", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    args, remaining = parser.parse_known_args()
    overrides = [r for r in remaining if "=" in r]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # ANSI helpers
    G = "\033[32m"
    R = "\033[31m"
    Y = "\033[33m"
    B = "\033[34m"
    C = "\033[36m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RST = "\033[0m"

    # ── Load training batch ──
    print(f"\n{BOLD}{C}📦 Loading training batch:{RST} {args.train_batch}")
    train_batch = torch.load(args.train_batch, map_location="cpu", weights_only=False)
    print(f"  {DIM}Keys:{RST} {sorted([k for k in train_batch if k != 'samples'])}")
    print(f"  {DIM}Batch size:{RST} {train_batch['action'].shape[0]}")
    train_sample_0 = train_batch["samples"][0]
    train_template = train_sample_0["template"]
    train_emb = _get_train_emb(train_batch)
    print(f"  {DIM}Embodiment:{RST} {BOLD}{train_emb}{RST}")
    print(f"  {DIM}Template:{RST} {repr(train_template[:100])}...")

    # ── Load config + processor ──
    run_dir = find_run_dir(args.ckpt_path)
    print(f"\n{BOLD}{C}📂 Run dir:{RST} {run_dir}")
    cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

    eval_embodiment = cfg.get("eval_embodiment", None)
    is_mixture = "embodiment_datasets" in cfg.data
    if is_mixture:
        if not eval_embodiment:
            first_emb = next(iter(cfg.data.embodiment_datasets))
            eval_embodiment = first_emb
            OmegaConf.set_struct(cfg, False)
            cfg.eval_embodiment = eval_embodiment
            print(f"Auto-selected embodiment: {eval_embodiment}")
        filter_embodiment(cfg, eval_embodiment)

    truncate_datasets(cfg, args.max_datasets)

    dataset_stats = load_dataset_stats_from_json(run_dir / "dataset_stats.json")
    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(dataset_stats)
    processor.eval()

    # ── Load dataset ──
    dataset_eval = instantiate(cfg.data, is_training_set=False)
    if isinstance(dataset_eval, MixtureLerobotDataset):
        emb = dataset_eval.embodiments[0]
        ds = dataset_eval.datasets[0]
        ds.processor = None
        emb_processor = processor[emb] if isinstance(processor, MixtureProcessor) else processor
        print(f"{BOLD}{C}🤖 Eval embodiment:{RST} {emb}")
    else:
        dataset_eval.processor = None
        ds = dataset_eval
        emb_processor = processor
        emb = eval_embodiment

    same_embodiment = train_emb == emb
    if not same_embodiment:
        print(
            f"\n  {Y}📝 NOTE:{RST} Training='{BOLD}{train_emb}{RST}', eval='{BOLD}{emb}{RST}' "
            f"— dim_pad_mask differences are expected."
        )

    # ── Get one sample and run both paths ──
    ds_sample = ds[0]
    raw_obs = dataset_sample_to_raw_obs(ds_sample, embodiment_type=emb)
    server_dict = build_obs_dict(raw_obs, processor)

    eval_preprocessed = emb_processor.preprocess(deepcopy(ds_sample))
    serve_preprocessed = emb_processor.preprocess(deepcopy(server_dict))

    # ══════════════════════════════════════════════════════════════════
    # Run all checks
    # ══════════════════════════════════════════════════════════════════
    report = ConsistencyReport()

    section_headers = [
        ("A", "🔤 Template checks", "critical because they directly affect input_ids"),
        ("B", "📐 Tensor shape checks", ""),
        ("C", "🎭 Dim pad mask checks", ""),
        ("D", "🤖 Embodiment field checks", ""),
        ("E", "🔀 Eval vs Serve value checks", ""),
        ("F", "🏗️  Structure checks", ""),
        ("G", "🔢 input_ids token checks", ""),
    ]

    def _print_section(idx: int):
        tag, title, note = section_headers[idx]
        note_str = f" {DIM}({note}){RST}" if note else ""
        print(f"\n{BOLD}{B}── {tag}. {title}{RST}{note_str}")

    # A: Template (most critical)
    _print_section(0)
    for name, preprocessed in [
        ("eval_open_loop", eval_preprocessed),
        ("serve_policy", serve_preprocessed),
    ]:
        s = _get_samples(preprocessed)
        other_template = s.get("template", "")
        if other_template:
            check_template(report, train_template, other_template, name)
        else:
            report.fail(f"A. Template: {name} has no template", "")

    # B: Tensor shapes
    _print_section(1)
    for name, preprocessed in [
        ("eval_open_loop", eval_preprocessed),
        ("serve_policy", serve_preprocessed),
    ]:
        check_tensor_shapes(report, train_batch, preprocessed, name)

    # C: Dim pad masks
    _print_section(2)
    for name, preprocessed in [
        ("eval_open_loop", eval_preprocessed),
        ("serve_policy", serve_preprocessed),
    ]:
        check_dim_pad_masks(report, train_batch, preprocessed, name, same_embodiment)

    # D: Embodiment fields
    _print_section(3)
    for name, preprocessed in [
        ("eval_open_loop", eval_preprocessed),
        ("serve_policy", serve_preprocessed),
    ]:
        check_embodiment_fields(report, train_batch, preprocessed, name)

    # E: Eval vs Serve value consistency
    _print_section(4)
    check_values_eval_vs_serve(report, eval_preprocessed, serve_preprocessed)

    # F: Structure completeness
    _print_section(5)
    for name, preprocessed in [
        ("eval_open_loop", eval_preprocessed),
        ("serve_policy", serve_preprocessed),
    ]:
        check_structure(report, train_batch, preprocessed, name)

    # G: input_ids token-level comparison
    _print_section(6)
    for name, preprocessed in [
        ("eval_open_loop", eval_preprocessed),
        ("serve_policy", serve_preprocessed),
    ]:
        check_input_ids_structure(report, train_batch, preprocessed, name)

    # ══════════════════════════════════════════════════════════════════
    # Print report
    # ══════════════════════════════════════════════════════════════════
    report.print_report()

    output_dir = Path(args.output_dir) if args.output_dir else Path(run_dir / "consistency_check")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(report.to_text())
    print(f"{DIM}💾 Report saved to:{RST} {report_path}")


if __name__ == "__main__":
    main()
