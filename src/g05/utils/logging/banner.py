"""Startup ASCII banner for entry scripts such as finetune / serve.

Does not use logging because RichHandler wraps lines to terminal width and adds
prefixes. Prints directly to stdout. In tty, applies a 256-color gradient per line;
when redirected to a file, outputs plain text.

Usage:
    from g05.utils.logging.banner import print_banner
    print_banner(subtitle="Post-Training")
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

_ART = r"""
   ______   ____    ______
  / ____/  / __ \  / ____/
 / / __   / / / / /___ \
/ /_/ /  / /_/ / ____/ /
\____/   \____(_)_____/
""".strip("\n")

_ART_WIDTH = max(len(line) for line in _ART.splitlines())
# Top-to-bottom 256-color gradient (cyan -> blue).
_GRADIENT = (51, 45, 39, 38, 33)
_RESET = "\x1b[0m"
_DIM = "\x1b[2m"


def _package_version() -> str:
    try:
        return "v" + version("g05")
    except PackageNotFoundError:
        return "dev"


def make_banner(subtitle: str = "", color: bool = False) -> str:
    """Build the banner string: ASCII art + centered tagline."""
    art_lines = _ART.splitlines()
    if color:
        art_lines = [
            f"\x1b[38;5;{_GRADIENT[min(i, len(_GRADIENT) - 1)]}m{line}{_RESET}"
            for i, line in enumerate(art_lines)
        ]
    tagline = " · ".join(
        part for part in ("Galaxea Foundation Model", _package_version(), subtitle) if part
    )
    width = max(_ART_WIDTH, len(tagline))
    rule = "─" * width
    footer = [rule, tagline.center(width).rstrip(), rule]
    if color:
        footer = [f"{_DIM}{line}{_RESET}" for line in footer]
    return "\n".join(["", *art_lines, *footer, ""])


def print_banner(subtitle: str = "") -> None:
    """Call once on the main process before all training/inference logs."""
    print(make_banner(subtitle, color=sys.stdout.isatty()), flush=True)
