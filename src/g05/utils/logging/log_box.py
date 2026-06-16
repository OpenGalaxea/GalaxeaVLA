"""Structured Unicode box logging utility.

Usage:
    from g05.utils.logging.log_box import log_box
    log_box(logger, "⚙  Run Configuration", [
        ("Output dir", "/path/to/output"),
        ("Checkpoint", "/path/to/ckpt.pt"),
        None,                                 # separator line
        ("World size", "2   Node rank: 0"),
    ])
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import IO, List, Optional, Union

Row = Union[str, tuple, None]

# ---------------------------------------------------------------------------
# Optional full-width file sink (set once from finetune.py / train entry points)
# ---------------------------------------------------------------------------
# When set, every log_box call appends a *non-truncated* (wide) version to this
# file in addition to the normal (possibly truncated) terminal output.
_log_file_path: Path | None = None
_log_file_handle: IO[str] | None = None


def set_log_file(path: "str | Path") -> None:
    """Configure a plain-text file that receives full-width (non-truncated) log_box output.

    Call once from the training entry point after output_dir is known.
    Thread-safety: only call from the main process before training starts.
    """
    global _log_file_path, _log_file_handle
    _log_file_path = Path(path)
    _log_file_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file_handle = open(_log_file_path, "a", encoding="utf-8")  # noqa: WPS515


def _write_to_file(text: str) -> None:
    """Append text to the log file if one has been configured."""
    if _log_file_handle is not None:
        try:
            _log_file_handle.write(text + "\n")
            _log_file_handle.flush()
        except Exception:  # noqa: BLE001
            pass


def make_box(title: str, rows: list[Row], inner_width: int = 62) -> str:
    """Build a Unicode box string.

    Args:
        title: title line text, centered
        rows: content rows, each item can be:
              - None        -> horizontal separator ╠═══╣
              - str         -> normal text row, truncated with ... if too long
              - (key, val)  -> key-value row with key left-aligned to 22 chars
        inner_width: content width excluding left/right ║ borders, default 62

    Returns:
        Multi-line string with newlines, ready for logger.info()
    """
    W = inner_width

    def _top() -> str:
        return "╔" + "═" * W + "╗"

    def _sep() -> str:
        return "╠" + "═" * W + "╣"

    def _bot() -> str:
        return "╚" + "═" * W + "╝"

    def _title_line(text: str) -> str:
        inner = f"  {text}  "
        if len(inner) > W:
            inner = inner[: W - 3] + "..."
        pad = W - len(inner)
        left = pad // 2
        right = pad - left
        return "║" + " " * left + inner + " " * right + "║"

    def _text_line(text: str) -> str:
        content = f"  {text}"
        if len(content) > W:
            content = content[: W - 3] + "..."
        return "║" + content.ljust(W) + "║"

    def _kv_line(key: str, val: str, key_w: int = 22) -> str:
        key_part = f"  {key:<{key_w}}"
        if len(key_part) > W:
            key_part = key_part[: W - 3] + "..."
        available = W - len(key_part)
        val_part = str(val)
        if len(val_part) > available:
            val_part = val_part[: max(0, available - 3)] + "..." if available > 3 else ""
        content = key_part + val_part
        return "║" + content.ljust(W) + "║"

    lines = [_top(), _title_line(title), _sep()]
    for row in rows:
        if row is None:
            lines.append(_sep())
        elif isinstance(row, tuple) and len(row) == 2:
            lines.append(_kv_line(str(row[0]), str(row[1])))
        else:
            lines.append(_text_line(str(row)))
    lines.append(_bot())
    return "\n".join(lines)


def log_box(
    logger: logging.Logger,
    title: str,
    rows: "List[Row]",
    inner_width: int = 62,
    file_rows: "Optional[List[Row]]" = None,
) -> None:
    """Emit a structured box with logger.info, prefixed by a blank visual spacer.

    Args:
        rows: terminal display rows, truncated by inner_width.
        file_rows: rows written to files without truncation, width 200. If None,
                   rows are reused. Useful when content already contains ellipses,
                   such as No-Decay mode lists.
    """
    logger.info("\n" + make_box(title, rows, inner_width))
    _write_to_file(
        "\n" + make_box(title, file_rows if file_rows is not None else rows, inner_width=200)
    )
