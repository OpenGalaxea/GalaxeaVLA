"""Client-side episode recorder: video + JSON + action curves.

Designed as a drop-in plugin for eval_open_loop client mode.
Collects images, GT actions, pred actions, and optional model outputs
(cot_text etc.) per step, then saves:
  - MP4 video (multi-camera concat + text overlay)
  - JSON metadata (per-step cot_text, task, timestamps)
  - Action plot PNG (GT vs pred curves, optional)

Usage in eval_open_loop::

    recorder = EpisodeRecorder(output_dir / "recordings")
    recorder.start(task="pick_and_place")
    for each_frame:
        recorder.add_step(
            images={"head_rgb": arr_CHW, ...},
            gt_action={"right_arm": arr_dim, ...},
            pred_action={"right_arm": arr_dim, ...},
            extra={"cot_text": "...", ...},
        )
    recorder.stop_and_save()
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Unicode / CJK text rendering ──

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]
_LATIN_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _load_cjk_font(font_size: int):
    """Try CJK font candidates in order; warn (not silently fail) if none found."""
    from PIL import ImageFont
    for path in _CJK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    logger.warning(
        "No CJK font found (tried: %s). Chinese text will render as tofu boxes.",
        _CJK_FONT_CANDIDATES,
    )
    return ImageFont.load_default()


def _puttext_unicode(roi: np.ndarray, lines: list[str], margin: int, line_height: int, font_size: int):
    """Draw multi-line Unicode text (including Chinese) on an RGB numpy image via PIL."""
    from PIL import Image, ImageFont, ImageDraw

    cjk_font = _load_cjk_font(font_size)
    try:
        latin_font = ImageFont.truetype(_LATIN_FONT_PATH, font_size)
    except Exception:
        latin_font = cjk_font

    pil_img = Image.fromarray(roi)
    draw = ImageDraw.Draw(pil_img)

    for i, line in enumerate(lines):
        y = margin + i * line_height
        x = margin
        # Split into CJK / non-CJK segments for correct font selection
        current, current_is_cjk = "", None
        segments = []
        for ch in line:
            ch_is_cjk = ord(ch) > 0x2E7F
            if current_is_cjk is None:
                current_is_cjk = ch_is_cjk
            if ch_is_cjk != current_is_cjk:
                if current:
                    segments.append((current, current_is_cjk))
                current, current_is_cjk = ch, ch_is_cjk
            else:
                current += ch
        if current:
            segments.append((current, current_is_cjk))

        for seg, is_cjk in segments:
            font = cjk_font if is_cjk else latin_font
            draw.text((x, y), seg, font=font, fill=(255, 255, 255))
            try:
                x += font.getlength(seg)
            except AttributeError:
                x += font.getsize(seg)[0]

    roi[:] = np.array(pil_img)


class EpisodeRecorder:
    """Record one episode at a time; call stop_and_save() between episodes."""

    def __init__(self, save_dir: str | Path, fps: int = 15):
        from datetime import datetime

        self.save_dir = Path(save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fps = fps
        self._steps: list[dict] = []
        self._active = False
        self._episode_idx = 0
        self._start_task = ""

    @property
    def active(self) -> bool:
        return self._active

    def start(self, task: str = ""):
        self._steps = []
        self._active = True
        self._start_task = task
        logger.info("Recording started: %s", task or "(no task)")

    def add_step(
        self,
        images: dict[str, np.ndarray] | None = None,
        gt_action: dict[str, np.ndarray] | None = None,
        pred_action: dict[str, np.ndarray] | None = None,
        extra: dict | None = None,
    ):
        """Append one step.

        Args:
            images: ``{cam_name: ndarray [C,H,W] uint8}``. None on cache-hit frames.
            gt_action: ``{part: ndarray [dim]}``.
            pred_action: ``{part: ndarray [dim]}``.
            extra: arbitrary dict (cot_text, task, etc.) — saved to JSON.
        """
        if not self._active:
            return
        step = {
            "timestamp": time.time(),
            "extra": extra or {},
        }
        if images is not None:
            step["images"] = {k: np.array(v, copy=True) for k, v in images.items()}
        else:
            step["images"] = None
        if gt_action is not None:
            step["gt_action"] = {k: np.array(v, copy=False) for k, v in gt_action.items()}
        if pred_action is not None:
            step["pred_action"] = {k: np.array(v, copy=False) for k, v in pred_action.items()}
        self._steps.append(step)

    def stop_and_save(self):
        if not self._active:
            return
        self._active = False
        if not self._steps:
            logger.info("Recording stopped (empty, not saved)")
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        tag = f"episode_{self._episode_idx:04d}"
        self._save_video(tag)
        self._save_json(tag)
        logger.info(
            "Episode %d saved: %d steps → %s/%s.*",
            self._episode_idx, len(self._steps), self.save_dir, tag,
        )
        self._episode_idx += 1
        self._steps = []

    # ── JSON ──

    def _save_json(self, tag: str):
        records = []
        for i, step in enumerate(self._steps):
            entry = {
                "step": i,
                "timestamp": step["timestamp"],
            }
            entry.update(step.get("extra", {}))
            if step["images"] is not None:
                entry["cameras"] = sorted(step["images"].keys())
            records.append(entry)
        path = self.save_dir / f"{tag}.json"
        path.write_text(json.dumps({
            "task": self._start_task,
            "num_steps": len(self._steps),
            "fps": self.fps,
            "steps": records,
        }, ensure_ascii=False, indent=2))

    # ── Video ──

    def _save_video(self, tag: str):
        import cv2
        import imageio

        # Find first step with images to get dimensions
        first_images = None
        for step in self._steps:
            if step["images"] is not None:
                first_images = step["images"]
                break
        if first_images is None:
            logger.warning("No images in recording, skipping video")
            return

        cam_keys = sorted(first_images.keys())
        # Use max height across cameras; resize others to match
        target_h = max(img.shape[1] for img in first_images.values())

        path = self.save_dir / f"{tag}.mp4"
        writer = imageio.get_writer(
            str(path), fps=self.fps, codec="libx264",
            quality=None, output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
        )

        last_frame = None
        for step in self._steps:
            if step["images"] is not None:
                # Build frame from images, resize to uniform height
                panels = []
                for cam in cam_keys:
                    img_chw = step["images"][cam]
                    img_rgb = np.transpose(img_chw, (1, 2, 0))
                    if img_rgb.shape[0] != target_h:
                        new_w = int(img_rgb.shape[1] * target_h / img_rgb.shape[0])
                        img_rgb = cv2.resize(img_rgb, (new_w, target_h))
                    panels.append(img_rgb)
                last_frame = np.concatenate(panels, axis=1)
            # else: reuse last_frame (cache-hit frame had no images)

            if last_frame is None:
                continue

            frame = last_frame.copy()
            overlay_text = step.get("extra", {}).get("cot_text", "")
            if overlay_text:
                self._overlay_text(frame, overlay_text)

            writer.append_data(frame)

        writer.close()
        logger.info("Video saved: %s (%d frames)", path, len(self._steps))

    @staticmethod
    def _overlay_text(frame: np.ndarray, text: str, max_width: int = 60):
        """Draw semi-transparent text overlay (RGB frame) supporting Chinese via PIL."""
        import cv2
        import textwrap

        if not text:
            return
        lines = []
        for paragraph in text.split("\n"):
            lines.extend(textwrap.wrap(paragraph, width=max_width) or [""])

        font_size = 24
        line_height = font_size + 6
        margin = 8
        text_h = margin + line_height * len(lines) + margin
        text_w = min(frame.shape[1], margin * 2 + max_width * (font_size // 2 + 2))

        # Semi-transparent black background
        overlay = frame[:text_h, :text_w].copy()
        cv2.rectangle(frame, (0, 0), (text_w, text_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame[:text_h, :text_w], 0.6, 0, frame[:text_h, :text_w])

        roi = frame[:text_h, :text_w]
        _puttext_unicode(roi, lines, margin, line_height, font_size)
