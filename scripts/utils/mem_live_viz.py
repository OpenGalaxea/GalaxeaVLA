"""Live CV2 visualizer for serve_policy_mem — per-camera windows + bbox overlay.

Simple by design: one cv2 window per camera, ``cv2.imshow`` + ``cv2.waitKey(1)``,
called synchronously from the websocket handler. No ``cv2.namedWindow`` (it can
hang on broken GTK/Qt backends; imshow creates the window lazily). COT text is
printed via logger, bbox rectangles are drawn on the head camera window.

BBox format from ``BBoxCoTBuilder._format_bbox_json``
(``src/g05/data_processor/samples_builder.py:414-434``)::

    "BBox: towel <loc0490><loc0115><loc0704><loc0286>; spoon <loc0200>..."

Each ``<locXXXX>`` is ``round(v * 1024)`` clamped to ``[0, 1023]``. Four tokens
per object, ordered ``y1, x1, y2, x2`` (PaliGemma yxyx). Coords are normalized
to ``[0, 1]``.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


_BBOX_RE = re.compile(r"([^;<>\n]+?)\s+<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>")
_AFFORD_RE = re.compile(
    r"(left_wrist|right_wrist|head)\s+camera:\s*\(\s*<loc(\d+)>\s*,\s*<loc(\d+)>\s*\)"
)

_BBOX_COLOR = (0, 200, 255)  # BGR yellow
_AFFORD_COLOR = (0, 255, 0)  # BGR green
_CACHE_COLOR = (0, 220, 255)  # BGR yellow

# ── CoT display helpers ──────────────────────────────────────────────────────
_LOC_STRIP_RE = re.compile(r"<loc\d+>")
_BBOX_FULL_STRIP_RE = re.compile(r"BBox\s*:[^.]*\.", re.IGNORECASE)
_AFFORD_STRIP_RE = re.compile(
    r"(?:left_wrist|right_wrist|head)\s+camera:\s*\([^)]*\)", re.IGNORECASE
)


def _clean_cot_for_display(cot_text: str) -> str:
    """Strip structured annotation tokens for human-readable overlay."""
    text = _BBOX_FULL_STRIP_RE.sub("", cot_text)
    text = _AFFORD_STRIP_RE.sub("", text)
    text = _LOC_STRIP_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _draw_cot_overlay(bgr: np.ndarray, cot_text: str) -> None:
    """Draw cleaned CoT text at the bottom of a BGR frame (modifies in-place)."""
    import cv2

    cleaned = _clean_cot_for_display(cot_text)
    if not cleaned:
        return
    h, w = bgr.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (_, lh), baseline = cv2.getTextSize("Ag", font, scale, thickness)
    line_stride = lh + baseline + 3
    margin = 5

    words = cleaned.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        cand = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(cand, font, scale, thickness)
        if tw > w - 2 * margin and cur:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        lines.append(cur)
    lines = lines[:4]  # cap at 4 lines to preserve camera view

    overlay_h = len(lines) * line_stride + 2 * margin
    y_start = max(0, h - overlay_h)
    roi = bgr[y_start:h, 0:w]
    cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0, dst=roi)
    y = y_start + margin + lh
    for line in lines:
        cv2.putText(bgr, line, (margin, y), font, scale, (180, 255, 180), thickness, cv2.LINE_AA)
        y += line_stride


def parse_bbox_cot(cot_text: str) -> list[tuple[str, float, float, float, float]]:
    """Extract ``[(name, x1, y1, x2, y2), ...]`` from COT, coords in ``[0, 1]``."""
    if not cot_text:
        return []
    out = []
    for m in _BBOX_RE.finditer(cot_text):
        name = m.group(1).strip().lstrip(":").strip()
        if name.lower().startswith("bbox"):
            name = name[4:].lstrip(":").strip()
        if not name:
            continue
        ly1, lx1, ly2, lx2 = (int(m.group(i)) for i in (2, 3, 4, 5))
        x1 = max(0.0, min(1.0, lx1 / 1024.0))
        y1 = max(0.0, min(1.0, ly1 / 1024.0))
        x2 = max(0.0, min(1.0, lx2 / 1024.0))
        y2 = max(0.0, min(1.0, ly2 / 1024.0))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        out.append((name, x1, y1, x2, y2))
    return out


def parse_affordance_cot(cot_text: str) -> list[tuple[str, float, float]]:
    """Legacy affordance UV format → ``[(view, u01, v01), ...]``."""
    if not cot_text:
        return []
    out = []
    for m in _AFFORD_RE.finditer(cot_text):
        v, lu, lv = m.group(1), int(m.group(2)), int(m.group(3))
        out.append((v, max(0.0, min(1.0, lu / 1024.0)), max(0.0, min(1.0, lv / 1024.0))))
    return out


def _to_hwc_bgr_u8(img) -> np.ndarray | None:
    """Coerce raw_obs camera frame to HWC uint8 BGR (ready for cv2.imshow)."""
    a = np.asarray(img)
    if a.ndim != 3:
        return None
    if a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (
            np.clip(a, 0, 255).astype(np.uint8)
            if a.max() > 1.5
            else (a * 255).clip(0, 255).astype(np.uint8)
        )
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] == 4:
        a = a[..., :3]
    # raw_obs frames are RGB; flip to BGR for cv2.
    return a[..., ::-1].copy()


class MemLiveVisualizer:
    """One cv2 window per camera. cot → logger; bbox → head window."""

    def __init__(self, window_prefix: str = "mem", target_panel_h: int = 360):
        self.window_prefix = window_prefix
        self.target_panel_h = target_panel_h
        self.enabled = True  # opt out lazily on first cv2 failure
        self._last: dict[str, np.ndarray] = {}  # cam_key -> last BGR frame
        self._windows: set[str] = set()
        self._last_cot: str | None = None  # persists across cache frames

    def update(
        self,
        images: dict | None,
        cot_text: str | None,
        is_fresh: bool,
        chunk_step: int,
        action_steps: int,
    ) -> bool:
        """Returns False if user pressed 'q' (server may ignore)."""
        if not self.enabled:
            return True
        try:
            import cv2

            if is_fresh:
                if cot_text is not None:
                    self._last_cot = cot_text
                    logger.info("[COT] %s", cot_text)
                if images:
                    bboxes = parse_bbox_cot(cot_text or "")
                    affords = parse_affordance_cot(cot_text or "")
                    for cam_key, img in images.items():
                        bgr = _to_hwc_bgr_u8(img)
                        if bgr is None or not bgr.any():
                            continue
                        h0, w0 = bgr.shape[:2]
                        if h0 != self.target_panel_h:
                            new_w = max(1, int(round(w0 * self.target_panel_h / h0)))
                            bgr = cv2.resize(bgr, (new_w, self.target_panel_h))
                        h, w = bgr.shape[:2]

                        # Bbox: main task camera only (head for MEM, exterior for SO100).
                        if "head" in cam_key or "exterior" in cam_key:
                            for name, x1, y1, x2, y2 in bboxes:
                                p1 = (int(round(x1 * w)), int(round(y1 * h)))
                                p2 = (int(round(x2 * w)), int(round(y2 * h)))
                                cv2.rectangle(bgr, p1, p2, _BBOX_COLOR, 2, cv2.LINE_AA)
                                cv2.putText(
                                    bgr,
                                    name,
                                    (p1[0] + 3, max(12, p1[1] - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    _BBOX_COLOR,
                                    1,
                                    cv2.LINE_AA,
                                )

                        # Legacy affordance UV: per-view single point.
                        for view, u01, v01 in affords:
                            if view not in cam_key:
                                continue
                            cx, cy = int(round(u01 * w)), int(round(v01 * h))
                            cv2.drawMarker(
                                bgr,
                                (cx, cy),
                                _AFFORD_COLOR,
                                cv2.MARKER_CROSS,
                                24,
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.circle(bgr, (cx, cy), 10, _AFFORD_COLOR, 1, cv2.LINE_AA)

                        self._last[cam_key] = bgr

            for cam_key, bgr in self._last.items():
                disp = bgr.copy()  # always copy — overlays below mutate
                if not is_fresh:
                    label = f"CACHE {chunk_step}/{action_steps}"
                    cv2.putText(
                        disp,
                        label,
                        (max(4, disp.shape[1] - 170), 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        _CACHE_COLOR,
                        2,
                        cv2.LINE_AA,
                    )
                if self._last_cot:
                    _draw_cot_overlay(disp, self._last_cot)
                win = f"{self.window_prefix}:{cam_key}"
                cv2.imshow(win, disp)
                self._windows.add(win)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("viz closed by user (q)")
                self.close()
                return False
        except Exception as exc:
            logger.warning("viz failed (%s); disabling further updates", exc)
            self.enabled = False
        return True

    def reset(self):
        """Drop cached last frames and CoT on episode boundary."""
        self._last.clear()
        self._last_cot = None

    def close(self):
        if not self.enabled and not self._windows:
            return
        self.enabled = False
        try:
            import cv2

            for win in self._windows:
                cv2.destroyWindow(win)
        except Exception:
            pass
        self._windows.clear()
