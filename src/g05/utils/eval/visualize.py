"""visualize.py — Eval visualization utilities.

Sections:
  1. Constants
  2. Internal helpers  (_make_gt_trace, _make_chunk_trace, _unnormalize_frame)
  3. Action plots — Plotly   (plot_result_ar_fm, plot_result_normalized, plot_result)
  4. Action plots — Matplotlib  (plot_chunks_matplotlib, _plot_normalized_matplotlib)
  5. Video plots  (plot_bbox, plot_subtask, plot_affordance)
  6. Unified entry point  (visualize_episode)
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import cv2 as cv
import imageio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


# =========================================================================
# 1. Constants
# =========================================================================

PALETTE = [
    "rgb(0, 255, 0)",
    "rgb(0, 191, 0)",
    "rgb(0, 128, 0)",
    "rgb(0, 64, 0)",
    "rgb(0, 0, 0)",
]

# 10 distinct colors for per-dim differentiation
DIM_COLORS = px.colors.qualitative.Plotly


# =========================================================================
# 2. Internal helpers
# =========================================================================


def _make_gt_trace(gt_1d, episode_size, color, name, legendgroup, showlegend):
    """Create a dashed GT line trace for a single action dimension."""
    return go.Scatter(
        x=np.arange(episode_size).tolist(),
        y=gt_1d.tolist(),
        mode="lines",
        line=dict(color=color, dash="dash", width=2),
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
    )


def _make_chunk_trace(
    chunks_1d,
    episode_size,
    chunk_size,
    color,
    name,
    legendgroup,
    showlegend,
    dash=None,
    subsample: int = 1,
    opacity: float = 0.3,
):
    """Create a semi-transparent merged-chunk trace for a single action dimension.

    Args:
        chunks_1d: ``(episode_size, chunk_size)`` — one dim sliced from the
            full ``(episode_size, chunk_size, dim)`` prediction tensor.
        subsample: draw one chunk every *subsample* frames (default 1 = all).
        opacity: trace opacity (default 0.3).
    """
    xs, ys = [], []
    if chunk_size == 1:
        # Single-step predictions: draw one continuous line instead of N dots
        for t in range(0, episode_size, subsample):
            xs.append(t)
            ys.append(float(chunks_1d[t, 0]))
        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=1.5, dash=dash),
            opacity=min(opacity + 0.3, 1.0),
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hoverinfo="skip",
        )
    for t in range(0, episode_size, subsample):
        xs.extend(np.arange(t, t + chunk_size).tolist())
        ys.extend(chunks_1d[t].tolist())
        xs.append(None)
        ys.append(None)
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=1, dash=dash),
        opacity=opacity,
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        hoverinfo="skip",
    )


def _unnormalize_frame(pixel_values_i):
    """Convert a single pixel_values frame to uint8 RGB numpy array.

    Handles both torch.Tensor and numpy inputs.  Assumes
    ``pixel_values_i`` has shape ``(1, C, H, W)`` with mean/std = 0.5.

    Returns:
        img: ``(H, W, 3)`` uint8 contiguous array, H, W ints.
    """
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    frame = pixel_values_i[0]  # (C, H, W)
    frame = frame * std[:, None, None] + mean[:, None, None]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    img = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    _, H, W = (
        pixel_values_i[0].shape
        if isinstance(pixel_values_i[0], np.ndarray)
        else pixel_values_i[0].shape
    )
    return img, H, W


# =========================================================================
# 3. Action plots — Plotly
# =========================================================================


def plot_result_ar_fm(
    path,
    gt: dict,
    fm_pd: dict = None,
    ar_pd: dict = None,
    filename: str = "result_ar_fm.html",
    chunk_subsample: int = 1,
    fm_label: str = "fm",
    ar_label: str = "ar",
    ar_opacity: float = 0.3,
    op_mask: dict = None,
):
    """Plot GT, AR prediction, and FM prediction grouped by body part.

    Each body part is one subplot row.  Within each subplot every action
    dimension gets a distinct color.  GT is dashed, FM chunks are solid
    semi-transparent, AR chunks (if provided) are dotted semi-transparent.

    When *fm_pd* is None (GT-only mode), only GT lines are drawn.

    A dropdown selector allows toggling between showing FM only, AR only,
    or both.

    Args:
        path: episode output directory. HTML saved as ``path/<filename>``.
        gt: ``{part_name: (episode_size, part_dim)}`` ground truth.
        fm_pd: ``{part_name: (episode_size, chunk_size, part_dim)}`` FM chunks.
            When *None*, only GT is plotted.
        ar_pd: optional ``{part_name: (episode_size, chunk_size, part_dim)}``
            AR chunks.  When *None* only GT and FM are plotted.
        filename: output HTML filename (default ``result_ar_fm.html``).
        chunk_subsample: draw one chunk every N frames (default 1 = all).
        fm_label: legend label prefix for FM traces (default ``"fm"``).
        ar_label: legend label prefix for AR traces (default ``"ar"``).
        ar_opacity: opacity for AR chunk traces (default 0.3).
    """
    parts = list(gt.keys())
    num_parts = len(parts)

    subplot_titles = [f"{p} ({gt[p].shape[-1]} dims)" for p in parts]
    fig = make_subplots(
        rows=num_parts,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02 if num_parts > 1 else 0.1,
    )

    # Track category ("gt" / "fm" / "ar") per trace for dropdown visibility
    trace_categories = []

    first_part = True
    for part_idx, part in enumerate(parts):
        row = part_idx + 1
        gt_part = gt[part]
        episode_size = gt_part.shape[0]
        part_dim = gt_part.shape[-1]

        fm_part = fm_pd[part] if fm_pd is not None else None
        if fm_part is not None:
            chunk_size = fm_part.shape[1]

        # Per-dim op_mask strip placed above the action curves.
        # Each dim gets its own horizontal lane; green = active (True), red = inactive (False).
        op_part = None
        if op_mask is not None and part in op_mask:
            op_part = np.asarray(op_mask[part]).astype(bool)
            if op_part.shape[0] != episode_size:
                op_part = None  # shape mismatch — skip
        if op_part is not None:
            gt_max = float(np.nanmax(gt_part))
            gt_min = float(np.nanmin(gt_part))
            gt_range = max(gt_max - gt_min, 1e-6)
            base_y = gt_max + 0.05 * gt_range
            lane_step = 0.04 * gt_range
            x_arr = np.arange(episode_size)
            for d in range(part_dim):
                mask_d = op_part[:, d] if op_part.ndim == 2 else op_part
                colors = np.where(mask_d, "#2ca02c", "#d62728").tolist()
                y_arr = np.full(episode_size, base_y + d * lane_step)
                fig.add_trace(
                    go.Scatter(
                        x=x_arr,
                        y=y_arr,
                        mode="markers",
                        marker=dict(color=colors, size=6, symbol="square"),
                        name=f"op_mask dim{d}",
                        legendgroup=f"op_mask dim{d}",
                        showlegend=first_part,
                        hovertemplate=f"dim{d}: %{{customdata}}<extra></extra>",
                        customdata=np.where(mask_d, "active", "inactive"),
                    ),
                    row=row,
                    col=1,
                )
                trace_categories.append("op_mask")

        for d in range(part_dim):
            color = DIM_COLORS[d % len(DIM_COLORS)]

            fig.add_trace(
                _make_gt_trace(
                    gt_part[:, d],
                    episode_size,
                    color,
                    name=f"gt dim{d}",
                    legendgroup=f"dim{d}",
                    showlegend=first_part,
                ),
                row=row,
                col=1,
            )
            trace_categories.append("gt")

            if fm_part is not None:
                fig.add_trace(
                    _make_chunk_trace(
                        fm_part[:, :, d],
                        episode_size,
                        chunk_size,
                        color,
                        name=f"{fm_label} dim{d}",
                        legendgroup=f"dim{d}",
                        showlegend=first_part,
                        subsample=chunk_subsample,
                    ),
                    row=row,
                    col=1,
                )
                trace_categories.append("fm")

            if ar_pd is not None:
                ar_part = ar_pd[part]
                fig.add_trace(
                    _make_chunk_trace(
                        ar_part[:, :, d],
                        episode_size,
                        chunk_size,
                        color,
                        name=f"{ar_label} dim{d}",
                        legendgroup=f"dim{d}",
                        showlegend=first_part,
                        dash="dot",
                        subsample=chunk_subsample,
                        opacity=ar_opacity,
                    ),
                    row=row,
                    col=1,
                )
                trace_categories.append("ar")

        first_part = False

    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="grey",
        spikedash="dot",
    )

    # --- Dropdown selector for FM / AR / Both ---
    if ar_pd is not None and fm_pd is not None:

        def _vis(show_fm, show_ar):
            out = []
            for cat in trace_categories:
                if cat == "gt" or cat == "op_mask":
                    out.append(True)
                elif cat == "fm":
                    out.append(show_fm)
                else:
                    out.append(show_ar)
            return out

        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.0,
                    xanchor="left",
                    y=1.12,
                    yanchor="top",
                    buttons=[
                        dict(
                            label=f"{fm_label.upper()} + {ar_label.upper()}",
                            method="update",
                            args=[{"visible": _vis(True, True)}],
                        ),
                        dict(
                            label=f"{fm_label.upper()} only",
                            method="update",
                            args=[{"visible": _vis(True, False)}],
                        ),
                        dict(
                            label=f"{ar_label.upper()} only",
                            method="update",
                            args=[{"visible": _vis(False, True)}],
                        ),
                    ],
                )
            ],
        )
        title = f"Action: gt (dashed) vs {ar_label} (dotted) vs {fm_label} (solid)"
    elif fm_pd is not None:
        title = f"Action: gt (dashed) vs {fm_label} (solid)"
    else:
        title = "Action: gt (dashed)"

    fig.update_layout(height=300 * num_parts, title_text=title, hovermode="x unified")
    fig.write_html(path / filename)
    print(f"Plot saved to {path / filename}")


def plot_result_normalized(path, gt, pd):
    """Plot normalized action per dim: one subplot per action dimension.

    Args:
        path: episode output directory. HTML saved as ``path/result_normalized.html``.
        gt: ``(episode_size, dim)`` normalized GT (chunk step 0).
        pd: ``(episode_size, chunk_size, dim)`` normalized predicted chunks.
    """
    episode_size, chunk_size, dim = pd.shape

    subplot_titles = [f"dim {d}" for d in range(dim)]
    fig = make_subplots(
        rows=dim,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=max(0.005, 0.02 / max(dim, 1)),
    )

    for d in range(dim):
        row = d + 1
        color = DIM_COLORS[d % len(DIM_COLORS)]
        fig.add_trace(
            _make_gt_trace(
                gt[:, d],
                episode_size,
                color,
                name=f"gt dim{d}",
                legendgroup="gt",
                showlegend=(d == 0),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            _make_chunk_trace(
                pd[:, :, d],
                episode_size,
                chunk_size,
                color,
                name=f"pd dim{d}",
                legendgroup="pd",
                showlegend=(d == 0),
            ),
            row=row,
            col=1,
        )

    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="grey",
        spikedash="dot",
    )
    fig.update_layout(
        height=150 * dim,
        title_text="Normalized action: gt (dashed) vs pred (solid)",
        hovermode="x unified",
    )
    fig.write_html(path / "result_normalized.html")
    print("Normalized plot saved to", path / "result_normalized.html")


def plot_result(path, gt, pd):
    """Legacy per-dim plotly visualization (one HTML per dimension)."""
    episode_size, chunk_size, dim = pd.shape
    for d in range(dim):
        fig = go.Figure()
        for t in range(episode_size):
            color_idx = t % len(PALETTE)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(t, t + chunk_size),
                    y=pd[t, :, d],
                    line=dict(color=PALETTE[color_idx]),
                    name=f"pd group {color_idx}",
                    legendgroup=f"pd group {color_idx}",
                    showlegend=t < len(PALETTE),
                )
            )
        fig.add_trace(
            go.Scatter(x=np.arange(episode_size), y=gt[:, d], name="gt", line=dict(color="red"))
        )
        fig.write_html(path / f"{d:02}.html")

    print("Result plot save to", path)


# =========================================================================
# 4. Action plots — Matplotlib
# =========================================================================


def plot_chunks_matplotlib(
    path,
    gt: dict,
    fm_pd: dict = None,
    ar_pd: dict = None,
    chunk_stride: int = 4,
    gt_state: dict = None,
):
    """Plot action chunks as matplotlib PNG: GT + FM + optional AR + optional state, one PNG per body part.

    For each body part, creates a vertical stack of subplots (one per action dim,
    then one per state dim if gt_state is provided).
    GT is a blue dashed line; FM/AR chunks are drawn every *chunk_stride* frames
    as semi-transparent lines. State dims are plotted as red solid lines.

    When *fm_pd* is None (GT-only mode), only GT action and state are drawn.

    Args:
        path: episode output directory. PNGs saved as ``path/result_chunks_{part}.png``.
        gt: ``{part_name: (episode_size, part_dim)}`` ground truth (step 0).
        fm_pd: ``{part_name: (episode_size, chunk_size, part_dim)}`` FM chunks.
            When *None*, only GT is plotted.
        ar_pd: optional ``{part_name: (episode_size, chunk_size, part_dim)}``
            AR chunks. None to skip.
        chunk_stride: draw one chunk every N frames (default 4).
        gt_state: optional ``{part_name: (episode_size, state_dim)}`` proprioceptive state.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = list(gt.keys())
    if not parts:
        return

    for part in parts:
        gt_p = gt[part]
        n_action_dims = gt_p.shape[-1]
        episode_size = gt_p.shape[0]

        fm_p = fm_pd[part] if fm_pd is not None else None
        chunk_size = fm_p.shape[1] if fm_p is not None else None
        ar_p = ar_pd[part] if ar_pd is not None else None

        # State dims for this part
        st_p = None
        if gt_state is not None and part in gt_state:
            st_p = gt_state[part]
            if st_p.ndim == 1:
                st_p = st_p[:, None]
            if np.abs(st_p).max() < 1e-8:
                st_p = None

        fig, axes = plt.subplots(
            n_action_dims,
            1,
            figsize=(16, 4 * n_action_dims),
            sharex=True,
            squeeze=False,
        )

        for d in range(n_action_dims):
            ax = axes[d, 0]

            # GT continuous line (left y-axis)
            x_full = np.arange(episode_size)
            ax.plot(
                x_full,
                gt_p[:, d],
                color="C0",
                linestyle="--",
                alpha=0.9,
                linewidth=2.5,
                label="GT action",
                zorder=10,
            )

            # FM chunks (or continuous line when chunk_size == 1)
            if fm_p is not None:
                if chunk_size == 1:
                    # Single-step: draw one continuous line
                    ax.plot(
                        x_full,
                        fm_p[:, 0, d],
                        color="C2",
                        alpha=0.65,
                        linewidth=1.5,
                        label="FM",
                        zorder=3,
                    )
                else:
                    fm_drawn = False
                    for t in range(0, episode_size, chunk_stride):
                        end = min(t + chunk_size, episode_size)
                        x_chunk = np.arange(t, end)
                        ax.plot(
                            x_chunk,
                            fm_p[t, : end - t, d],
                            color="C2",
                            alpha=0.35,
                            linewidth=1.5,
                            label="FM" if not fm_drawn else None,
                            zorder=3,
                        )
                        ax.plot(
                            t,
                            fm_p[t, 0, d],
                            marker="o",
                            color="C2",
                            markersize=3,
                            alpha=0.6,
                            zorder=4,
                        )
                        fm_drawn = True

            # AR chunks (or continuous line when chunk_size == 1)
            if ar_p is not None:
                ar_chunk_size = ar_p.shape[1] if ar_p is not None else chunk_size
                if ar_chunk_size == 1:
                    ax.plot(
                        x_full,
                        ar_p[:, 0, d],
                        color="C1",
                        alpha=0.65,
                        linewidth=1.5,
                        label="AR",
                        zorder=3,
                    )
                else:
                    ar_drawn = False
                    for t in range(0, episode_size, chunk_stride):
                        end = min(t + chunk_size, episode_size)
                        x_chunk = np.arange(t, end)
                        ax.plot(
                            x_chunk,
                            ar_p[t, : end - t, d],
                            color="C1",
                            alpha=0.35,
                            linewidth=1.5,
                            label="AR" if not ar_drawn else None,
                            zorder=3,
                        )
                        ax.plot(
                            t,
                            ar_p[t, 0, d],
                            marker="o",
                            color="C1",
                            markersize=3,
                            alpha=0.6,
                            zorder=4,
                        )
                        ar_drawn = True

            ax.set_ylabel("action / state", fontsize=9)
            ax.tick_params(axis="y")
            ax.grid(True, alpha=0.3)

            # State on same y-axis (shared scale for direct comparison)
            if st_p is not None and d < st_p.shape[-1]:
                ax.plot(
                    x_full,
                    st_p[:episode_size, d],
                    color="C3",
                    alpha=0.7,
                    linewidth=1.5,
                    label="state",
                    zorder=2,
                )

            if d == 0:
                ax.legend(fontsize=9, loc="upper right")

            ax.set_title(f"dim {d}", fontsize=11)

        mode_str = "GT action" + (" + state" if st_p is not None else "")
        if fm_p is not None:
            mode_str = f"action chunks (stride={chunk_stride})" + (
                " + state" if st_p is not None else ""
            )
        fig.suptitle(f"{part} — {mode_str}", fontsize=14)
        fig.tight_layout()
        output_file = path / f"result_chunks_{part}.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Chunk plot saved to {output_file}")


def _plot_normalized_matplotlib(path, gt, pd):
    """Plot normalized GT vs pred step-0 as matplotlib PNG.

    Args:
        path: episode directory. PNG saved as ``path/result.png``.
        gt: ``(ep_len, D)`` normalized GT step 0.
        pd: ``(ep_len, D)`` normalized pred step 0.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep_len, num_dims = gt.shape
    n_cols = 4
    n_rows = max(1, (num_dims + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, squeeze=False
    )

    for d in range(num_dims):
        ax = axes[d // n_cols, d % n_cols]
        ax.plot(
            range(ep_len), gt[:, d], color="C0", linestyle="--", alpha=0.8, linewidth=1, label="GT"
        )
        ax.plot(
            range(ep_len), pd[:, d], color="C1", linestyle="-", alpha=0.7, linewidth=1, label="Pred"
        )
        ax.set_title(f"dim {d}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7)

    for d in range(num_dims, n_rows * n_cols):
        axes[d // n_cols, d % n_cols].set_visible(False)

    fig.suptitle("Normalized action (step 0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path / "result.png", dpi=120)
    plt.close(fig)


# =========================================================================
# 5. Video plots
# =========================================================================


def plot_bbox(path, pixel_values, pred_bbox, gt_bbox, fps: int = 15):
    """Render bbox video: GT (green) vs pred (blue) bounding boxes.

    Args:
        path: output video file path (e.g. ``episode_dir / "bbox.mp4"``).
        pixel_values: ``(T, 1, C, H, W)`` image tensor.
        pred_bbox / gt_bbox: list of ``<loc>`` strings, length T.
    """
    gt_color = (0, 255, 0)  # BGR
    pd_color = (0, 0, 255)  # BGR
    box_th = 2

    image_num = pixel_values.shape[0]
    if len(pred_bbox) != image_num or len(gt_bbox) != image_num:
        raise ValueError(
            f"Length mismatch: T={image_num}, pred={len(pred_bbox)}, gt={len(gt_bbox)}"
        )

    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None
    )

    def parse_box(text):
        locs = re.findall(r"<loc(\d+)>", "" if text is None else str(text))
        if len(locs) < 4:
            return None
        ymin, xmin, ymax, xmax = map(int, locs[:4])
        ymin = int(np.clip(ymin, 0, 1024))
        xmin = int(np.clip(xmin, 0, 1024))
        ymax = int(np.clip(ymax, 0, 1024))
        xmax = int(np.clip(xmax, 0, 1024))
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax == ymin or xmax == xmin:
            return None
        return ymin, xmin, ymax, xmax

    def to_xyxy(box, H, W):
        ymin, xmin, ymax, xmax = box
        x1 = int(np.clip(int(xmin / 1024.0 * W), 0, W - 1))
        x2 = int(np.clip(int(xmax / 1024.0 * W), 0, W - 1))
        y1 = int(np.clip(int(ymin / 1024.0 * H), 0, H - 1))
        y2 = int(np.clip(int(ymax / 1024.0 * H), 0, H - 1))
        return x1, y1, x2, y2

    for i in range(image_num):
        img, H, W = _unnormalize_frame(pixel_values[i])

        for text, color in [(gt_bbox[i], gt_color), (pred_bbox[i], pd_color)]:
            box = parse_box(text)
            if box is not None:
                x1, y1, x2, y2 = to_xyxy(box, H, W)
                cv.rectangle(img, (x1, y1), (x2, y2), color, box_th)

        writer.append_data(img)

    writer.close()
    print(f"BBox video saved to {path}")


_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]
_LATIN_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_UNICODE_FONT_SIZE = 16


def _load_cjk_font(font_size: int):
    """Try CJK font candidates in order; warn (not silently fail) if none found."""
    from PIL import ImageFont
    import logging as _logging

    for path in _CJK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    _logging.getLogger(__name__).warning(
        "No CJK font found (tried: %s). Chinese text will render as tofu boxes.",
        _CJK_FONT_CANDIDATES,
    )
    return ImageFont.load_default()


def _puttext_unicode(
    img: np.ndarray, text: str, pos, color_rgb, font_size: int = _UNICODE_FONT_SIZE
):
    """Draw Unicode text (including Chinese) on an RGB numpy image via PIL.

    Uses DejaVuSans for Latin/ASCII and a CJK font for Chinese characters,
    rendering each segment with the appropriate font to avoid tofu boxes.
    """
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    cjk_font = _load_cjk_font(font_size)
    try:
        latin_font = ImageFont.truetype(_LATIN_FONT_PATH, font_size)
    except Exception:
        latin_font = cjk_font

    # Split text into CJK and non-CJK segments and render each with the right font
    segments = []
    current = ""
    current_is_cjk = None
    for ch in text:
        ch_is_cjk = ord(ch) > 0x2E7F
        if current_is_cjk is None:
            current_is_cjk = ch_is_cjk
        if ch_is_cjk != current_is_cjk:
            if current:
                segments.append((current, current_is_cjk))
            current = ch
            current_is_cjk = ch_is_cjk
        else:
            current += ch
    if current:
        segments.append((current, current_is_cjk))

    x, y = pos
    for seg, is_cjk in segments:
        font = cjk_font if is_cjk else latin_font
        draw.text((x, y), seg, font=font, fill=color_rgb)
        try:
            x += font.getlength(seg)
        except AttributeError:
            x += font.getsize(seg)[0]  # Pillow < 8 fallback

    img[:] = np.array(pil_img)


def plot_subtask(path, pixel_values, pred_subtask, gt_subtask, fps: int = 15, output_size=None):
    """Render subtask text overlay video: GT (green) vs pred (blue).

    Args:
        path: output video file path (e.g. ``episode_dir / "subtask.mp4"``).
        pixel_values: ``(T, 1, C, H, W)`` image tensor.
        pred_subtask / gt_subtask: list of text strings, length T.
        output_size: optional ``(W, H)`` tuple to resize frames before rendering.
            Useful when pixel_values are VLM-preprocessed (e.g. 224×224) but you
            want the output at the original camera resolution (e.g. 1280×720).
    """
    gt_color_rgb = (0, 255, 0)
    pred_color_rgb = (0, 0, 255)

    image_num = pixel_values.shape[0]
    if len(pred_subtask) != image_num or len(gt_subtask) != image_num:
        raise ValueError(
            f"Length mismatch: T={image_num}, pred={len(pred_subtask)}, gt={len(gt_subtask)}"
        )

    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None
    )

    for i in range(image_num):
        img, H, W = _unnormalize_frame(pixel_values[i])

        if output_size is not None:
            img = cv.resize(img, output_size, interpolation=cv.INTER_LINEAR)
            W, H = output_size

        panel_h = min(H, 80)
        overlay = img.copy()
        cv.rectangle(overlay, (0, 0), (W, panel_h), (0, 0, 0), -1)
        img = cv.addWeighted(overlay, 0.45, img, 0.55, 0)

        _puttext_unicode(img, f"GT : {gt_subtask[i]}", (10, 8), gt_color_rgb)
        _puttext_unicode(img, f"PD : {pred_subtask[i]}", (10, 38), pred_color_rgb)

        writer.append_data(img)

    writer.close()
    print(f"Subtask video saved to {path}")


def plot_affordance(path, pixel_values, pred_affordance, gt_affordance):
    """Render affordance point video: GT (green) vs pred (red).

    Args:
        path: output directory. Video saved as ``path/affordance.mp4``.
        pixel_values: ``(T, 1, C, H, W)`` image tensor.
        pred_affordance / gt_affordance: list of ``head camera: (<loc>)`` strings, length T.
    """
    gt_color = (0, 255, 0)
    pred_color = (255, 0, 0)

    image_num = pixel_values.shape[0]
    video_path = path / "affordance.mp4"

    path.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        video_path, fps=15, codec="libx264", pixelformat="yuv420p", macro_block_size=None
    )

    def draw_point(image, text_data, color, radius, H, W):
        head_match = re.search(r"head camera:\s*\((.*?)\)", text_data)
        if head_match is None:
            return
        locs = re.findall(r"<loc(\d+)>", head_match.group(1))
        if len(locs) >= 2:
            real_x = int(np.clip(int(locs[0]) / 1024 * W, 0, W - 1))
            real_y = int(np.clip(int(locs[1]) / 1024 * H, 0, H - 1))
            cv.circle(image, (real_x, real_y), radius, color, -1)

    for i in range(image_num):
        img, H, W = _unnormalize_frame(pixel_values[i])
        draw_point(img, gt_affordance[i], gt_color, radius=5, H=H, W=W)
        draw_point(img, pred_affordance[i], pred_color, radius=2, H=H, W=W)
        writer.append_data(img)

    writer.close()
    print(f"Affordance video saved to {video_path}")


# =========================================================================
# 6. Unified entry point
# =========================================================================


def visualize_episode(
    episode_dir,
    gt_step0,
    pred_chunks,
    secondary_chunks=None,
    raw_gt_step0=None,
    raw_pred_chunks=None,
    primary_label="fm",
    secondary_label="ar",
    html_filename="result_ar_fm.html",
    chunk_subsample=1,
    secondary_opacity=0.3,
    extra_pkl_data=None,
    gt_state=None,
    op_mask=None,
):
    """Unified single-episode visualization entry point.

    Generates all visualization artifacts for one episode:
      1. Per-body-part interactive HTML (plotly)   → ``{html_filename}``
      2. Per-body-part static PNG (matplotlib)     → ``result_chunks.png``
      3. Normalized per-dim interactive HTML        → ``result_normalized.html``
      4. Normalized step-0 static PNG               → ``result.png``
      5. Data export                                → ``actions.pkl``

    Steps 3-4 are skipped when *raw_gt_step0* or *raw_pred_chunks* is None.

    Args:
        episode_dir: output directory for this episode (Path).
        gt_step0: ``{part: (ep_len, part_dim)}`` denormalized GT at chunk step 0.
        pred_chunks: ``{part: (ep_len, chunk, part_dim)}`` denormalized primary predictions.
        secondary_chunks: optional ``{part: (ep_len, chunk, part_dim)}`` secondary predictions.
        raw_gt_step0: ``(ep_len, valid_D)`` normalized GT step 0 (pad dims removed).
        raw_pred_chunks: ``(ep_len, T, valid_D)`` normalized pred chunks (pad dims removed).
        primary_label: legend label for primary predictions (default ``"fm"``).
        secondary_label: legend label for secondary predictions (default ``"ar"``).
        html_filename: output HTML filename (default ``"result_ar_fm.html"``).
        chunk_subsample: draw one chunk every N frames (default 1).
        secondary_opacity: opacity for secondary chunk traces (default 0.3).
        extra_pkl_data: additional data to include in actions.pkl.
    """
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-body-part interactive HTML
    plot_result_ar_fm(
        episode_dir,
        gt_step0,
        fm_pd=pred_chunks,
        ar_pd=secondary_chunks,
        filename=html_filename,
        chunk_subsample=chunk_subsample,
        fm_label=primary_label,
        ar_label=secondary_label,
        ar_opacity=secondary_opacity,
        op_mask=op_mask,
    )

    # 2. Per-body-part static PNG (action + state in same figure)
    if gt_state is not None:
        print(
            f"[DEBUG] action parts: {list(gt_step0.keys())}, state parts: {list(gt_state.keys())}"
        )
    plot_chunks_matplotlib(
        episode_dir,
        gt_step0,
        fm_pd=pred_chunks,
        ar_pd=secondary_chunks,
        gt_state=gt_state,
    )

    # 5. PKL export
    pkl_data = {
        "gt": gt_step0,
    }
    if pred_chunks is not None:
        pkl_data["fm_pd"] = pred_chunks
    if raw_gt_step0 is not None:
        pkl_data["raw_gt"] = raw_gt_step0
    if raw_pred_chunks is not None:
        pkl_data["raw_pd"] = raw_pred_chunks
    if extra_pkl_data:
        pkl_data.update(extra_pkl_data)

    with open(episode_dir / "actions.pkl", "wb") as f:
        pickle.dump(pkl_data, f)


# =========================================================================

if __name__ == "__main__":
    plot_affordance(
        Path("/vla_fulltime/xiao.liu/"),
        pixel_values=np.random.rand(100, 3, 3, 224, 224),
        pred_affordance=["head camera: (<loc512><loc512>)"] * 100,
        gt_affordance=["head camera: (<loc256><loc256>)"] * 100,
    )
