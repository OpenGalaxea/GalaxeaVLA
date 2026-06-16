"""Simple cv2 visualizer for camera frames with CoT bbox/text overlays."""
import logging
import re
import numpy as np

logger = logging.getLogger(__name__)

_BBOX_RE = re.compile(r"([^;<>\n]+?)\s+<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>")
# strip action tokens and loc tokens for readable display
_TOKEN_RE = re.compile(r"<[^>]+>")
_BBOX_COLOR  = (0, 200, 255)   # BGR yellow-orange
_TEXT_COLOR  = (220, 220, 220)
_TEXT_BG     = (30, 30, 30)
_PANEL_H     = 360
_TEXT_PANEL_H = 80             # height of COT text strip below image
_FONT        = cv2 = None      # lazy import


def _readable_cot(cot_text: str) -> str:
    """Strip action/loc tokens and keep human-readable segments."""
    clean = _TOKEN_RE.sub("", cot_text)
    # collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _wrap_text(text: str, font, font_scale: float, thickness: int, max_w: int) -> list[str]:
    """Word-wrap text to fit max_w pixels wide."""
    import cv2
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if tw <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def show(images: dict, cot_text: str | None) -> bool:
    """Draw images with bbox + COT text overlay. Returns False if user pressed 'q'."""
    import cv2

    for cam_key, img in images.items():
        a = np.asarray(img)
        if a.ndim != 3:
            continue
        if a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
            a = np.transpose(a, (1, 2, 0))
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        bgr = a[..., ::-1].copy()   # RGB → BGR

        h0, w0 = bgr.shape[:2]
        if h0 != _PANEL_H:
            new_w = max(1, int(round(w0 * _PANEL_H / h0)))
            bgr = cv2.resize(bgr, (new_w, _PANEL_H))
        h, w = bgr.shape[:2]

        is_head = "head" in cam_key

        # ── bbox rectangles (head camera only) ──────────────────────────────
        if is_head and cot_text:
            for m in _BBOX_RE.finditer(cot_text):
                name = m.group(1).strip().lstrip(":").strip()
                if name.lower().startswith("bbox"):
                    name = name[4:].lstrip(":").strip()
                ly1, lx1, ly2, lx2 = (int(m.group(i)) for i in (2, 3, 4, 5))
                x1 = max(0, int(lx1 / 1024 * w))
                y1 = max(0, int(ly1 / 1024 * h))
                x2 = min(w, int(lx2 / 1024 * w))
                y2 = min(h, int(ly2 / 1024 * h))
                cv2.rectangle(bgr, (x1, y1), (x2, y2), _BBOX_COLOR, 2, cv2.LINE_AA)
                cv2.putText(bgr, name, (x1 + 3, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BBOX_COLOR, 1, cv2.LINE_AA)

        # ── CACHE badge when no fresh cot ───────────────────────────────────
        if not cot_text:
            cv2.putText(bgr, "CACHE", (max(4, w - 100), 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

        # ── COT text panel below the image (head camera only) ───────────────
        if is_head:
            font       = cv2.FONT_HERSHEY_SIMPLEX
            fscale     = 0.45
            thickness  = 1
            line_h     = 18
            pad        = 6

            text_panel = np.full((_TEXT_PANEL_H, w, 3), _TEXT_BG, dtype=np.uint8)

            if cot_text:
                readable = _readable_cot(cot_text)
                lines = _wrap_text(readable, font, fscale, thickness, w - 2 * pad)
            else:
                lines = ["(waiting for next inference...)"]

            for i, line in enumerate(lines):
                y = pad + (i + 1) * line_h
                if y > _TEXT_PANEL_H - pad:
                    break
                cv2.putText(text_panel, line, (pad, y), font, fscale,
                            _TEXT_COLOR, thickness, cv2.LINE_AA)

            bgr = np.vstack([bgr, text_panel])

        cv2.imshow(f"efm:{cam_key}", bgr)

    return (cv2.waitKey(1) & 0xFF) != ord("q")


def close():
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass
