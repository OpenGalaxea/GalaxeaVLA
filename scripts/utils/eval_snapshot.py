import base64
import io
import logging
import math
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _is_finite(v):
    try:
        return math.isfinite(v)
    except (TypeError, ValueError, OverflowError):
        return False


def _sanitize_nested(data):
    if data is None:
        return None
    if isinstance(data, list):
        return [_sanitize_nested(item) for item in data]
    if isinstance(data, (int, float)):
        return data if _is_finite(data) else None
    return data


def _to_list(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    try:
        raw = x.tolist() if hasattr(x, "tolist") else list(x)
    except Exception:
        return None
    return _sanitize_nested(raw)


_DIM_COLORS = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def _tensor_to_base64_image(tensor_chw):
    import torchvision.transforms.functional as F

    frame = tensor_chw * 0.5 + 0.5
    img = F.to_pil_image(frame.clamp(0, 1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _get_batch_size(batch):
    pv = batch.get("pixel_values")
    if pv is None:
        action = batch.get("action")
        if isinstance(action, torch.Tensor):
            return action.shape[0]
        return 0
    if isinstance(pv, dict):
        if not pv:
            action = batch.get("action")
            if isinstance(action, torch.Tensor):
                return action.shape[0]
            return 0
        return next(iter(pv.values())).shape[0]
    return pv.shape[0]


def _render_images_for_sample(batch, sample_idx, camera_names=None):
    pv = batch.get("pixel_values")
    if pv is None:
        return ""

    imgs = []
    if isinstance(pv, dict):
        for cam_key in sorted(pv.keys()):
            cam_tensor = pv[cam_key][sample_idx]
            n_obs = cam_tensor.shape[0]
            for obs_idx in range(n_obs):
                try:
                    b64 = _tensor_to_base64_image(cam_tensor[obs_idx])
                    label = f"{cam_key}" if n_obs == 1 else f"{cam_key}[{obs_idx}]"
                    imgs.append(
                        f'<div class="cam-cell"><div class="cam-label">{label}</div>'
                        f'<img src="data:image/jpeg;base64,{b64}" '
                        f'alt="{label}" title="{label}"/></div>'
                    )
                except Exception:
                    imgs.append(f"<div><em>{cam_key}: decode error</em></div>")
    else:
        cam_tensor = pv[sample_idx]
        n_cam = cam_tensor.shape[0]
        labels = camera_names or [f"cam{j}" for j in range(n_cam)]
        for c in range(n_cam):
            try:
                b64 = _tensor_to_base64_image(cam_tensor[c])
                label = labels[c] if c < len(labels) else f"cam{c}"
                imgs.append(
                    f'<div class="cam-cell"><div class="cam-label">{label}</div>'
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    f'alt="{label}" title="{label}"/></div>'
                )
            except Exception:
                imgs.append(f"<div><em>cam{c}: decode error</em></div>")

    if not imgs:
        return ""
    return '<div class="images-row">' + "\n".join(imgs) + "</div>"


def _fmt_vec(vals, precision=4):
    parts = []
    for v in vals:
        try:
            if math.isfinite(v):
                parts.append(f"{v:.{precision}f}")
            else:
                parts.append(str(v))
        except (TypeError, ValueError, OverflowError):
            parts.append("?")
    return ", ".join(parts)


def _split_by_parts(action, parts_meta):
    if action is None or not parts_meta:
        return None
    result = {}
    offset = 0
    for name, dim in parts_meta.items():
        result[name] = [row[offset : offset + dim] for row in action]
        offset += dim
    return result


def _svg_path_str(x_vals, y_vals, x_scale, y_scale, y_min, plot_bottom, pad_l):
    parts = []
    in_segment = False
    for x, y in zip(x_vals, y_vals):
        if y is None:
            in_segment = False
            continue
        sx = pad_l + x * x_scale
        sy = plot_bottom - (y - y_min) * y_scale
        if not in_segment:
            parts.append(f"M{sx:.2f},{sy:.2f}")
            in_segment = True
        else:
            parts.append(f"L{sx:.2f},{sy:.2f}")
    return " ".join(parts)


def _series_label(has_pred, has_ar):
    labels = ["GT"]
    if has_pred:
        labels.append("FM")
    if has_ar:
        labels.append("AR")
    return "/".join(labels)


def _build_dim_focus_controls(uid, pidx, dim, has_pred, has_ar):
    group = f"dim-{uid}-{pidx}"
    all_id = f"{group}-all"
    inputs = [f'<input type="radio" name="{group}" id="{all_id}" class="dim-focus-radio" checked/>']
    labels = [f'<label for="{all_id}" class="dim-focus-label dim-all">All</label>']
    css = [f"#{all_id}:checked ~ .dim-tabs .dim-all{{background:#4A90D9;color:#fff;opacity:1;}}"]
    series = _series_label(has_pred, has_ar)

    for d in range(dim):
        did = f"{group}-{d}"
        color = _DIM_COLORS[d % len(_DIM_COLORS)]
        inputs.append(f'<input type="radio" name="{group}" id="{did}" class="dim-focus-radio"/>')

        swatches = [f'<span class="legend-swatch swatch-gt" style="border-color:{color}"></span>']
        if has_pred:
            swatches.append(
                f'<span class="legend-swatch swatch-pred" style="border-color:{color}"></span>'
            )
        if has_ar:
            swatches.append(
                f'<span class="legend-swatch swatch-ar" style="border-color:{color}"></span>'
            )
        labels.append(
            f'<label for="{did}" class="dim-focus-label dim-tab-{d}" '
            f'data-dim-input="{did}" data-all-input="{all_id}">'
            f"{''.join(swatches)}<span>{series} d{d}</span></label>"
        )
        css.append(
            f"#{did}:checked ~ .dim-tabs .dim-focus-label"
            "{opacity:0.28;}"
            f"#{did}:checked ~ .dim-tabs .dim-tab-{d}"
            "{background:#4A90D9;color:#fff;opacity:1;}"
            f"#{did}:checked ~ .chart-wrap .dim-line"
            "{opacity:0.08;}"
            f"#{did}:checked ~ .chart-wrap .dim-line.dim-d{d}"
            "{opacity:1;}"
        )

    return (
        "".join(inputs) + '<div class="dim-tabs">' + "".join(labels) + "</div>",
        "".join(css),
    )


def _build_action_plot_html(
    action_gt,
    action_pred,
    ar_pred,
    parts_meta,
    section_label="",
    plot_counter=[0],
):
    has_gt = action_gt is not None
    has_pred = action_pred is not None
    has_ar = ar_pred is not None
    if not has_gt:
        return ""

    action_gt = _to_list(action_gt)
    action_pred = _to_list(action_pred) if has_pred else None
    ar_pred = _to_list(ar_pred) if has_ar else None

    T = len(action_gt)
    if T == 0:
        return ""

    if parts_meta:
        gt_parts = _split_by_parts(action_gt, parts_meta)
        pred_parts = _split_by_parts(action_pred, parts_meta) if has_pred else None
        ar_pred_parts = _split_by_parts(ar_pred, parts_meta) if has_ar else None
        part_names = list(parts_meta.keys())
        part_dims = {name: parts_meta[name] for name in part_names}
    else:
        D = len(action_gt[0]) if action_gt[0] else 0
        gt_parts = {"action": action_gt}
        pred_parts = {"action": action_pred} if has_pred else None
        ar_pred_parts = {"action": ar_pred} if has_ar else None
        part_names = ["action"]
        part_dims = {"action": D}

    plot_counter[0] += 1
    uid = plot_counter[0]

    pad_l, pad_r, pad_t, pad_b = 55, 20, 20, 30
    svg_w = 740
    plot_w = svg_w - pad_l - pad_r
    plot_h = 180
    svg_h = pad_t + plot_h + pad_b

    radio_inputs = []
    tab_labels = []
    svg_panels = []

    for pidx, pname in enumerate(part_names):
        dim = part_dims[pname]
        gt_p = gt_parts[pname]
        pred_p = pred_parts[pname] if pred_parts else None
        ar_p = ar_pred_parts[pname] if ar_pred_parts else None

        all_vals = []
        for d in range(dim):
            all_vals.extend(v for row in gt_p for v in [row[d]] if v is not None)
            if pred_p:
                all_vals.extend(v for row in pred_p for v in [row[d]] if v is not None)
            if ar_p:
                all_vals.extend(v for row in ar_p for v in [row[d]] if v is not None)

        finite_vals = [v for v in all_vals if math.isfinite(v)]
        if not finite_vals:
            continue

        y_min = min(finite_vals)
        y_max = max(finite_vals)
        y_range = y_max - y_min
        if y_range < 1e-8:
            y_range = 1.0
            y_min -= 0.5
            y_max += 0.5
        y_pad = y_range * 0.08
        y_min -= y_pad
        y_max += y_pad
        y_range = y_max - y_min
        y_scale = plot_h / y_range
        x_scale = plot_w / max(T - 1, 1)
        plot_bottom = pad_t + plot_h

        y_ticks = _nice_ticks(y_min, y_max, 5)
        grid_lines = ""
        tick_labels = ""
        for yt in y_ticks:
            sy = plot_bottom - (yt - y_min) * y_scale
            if sy < pad_t - 2 or sy > plot_bottom + 2:
                continue
            grid_lines += f'<line x1="{pad_l}" y1="{sy:.1f}" x2="{pad_l + plot_w}" y2="{sy:.1f}" class="grid-line"/>'
            tick_labels += f'<text x="{pad_l - 6}" y="{sy:.1f}" class="tick-label" dominant-baseline="middle">{yt:.4g}</text>'

        x_ticks = _nice_ticks(0, T - 1, min(8, T))
        x_tick_labels = ""
        for xt in x_ticks:
            sx = pad_l + xt * x_scale
            x_tick_labels += f'<text x="{sx:.1f}" y="{plot_bottom + 18}" class="tick-label" text-anchor="middle">{int(xt)}</text>'

        paths = ""
        for d in range(dim):
            color = _DIM_COLORS[d % len(_DIM_COLORS)]
            gt_y = [row[d] for row in gt_p]

            paths += (
                f'<path d="{_svg_path_str(list(range(T)), gt_y, x_scale, y_scale, y_min, plot_bottom, pad_l)}" '
                f'class="line-gt dim-line dim-d{d}" data-dim="{d}" stroke="{color}"/>'
            )

            if pred_p is not None:
                pred_y = [row[d] for row in pred_p]
                paths += (
                    f'<path d="{_svg_path_str(list(range(T)), pred_y, x_scale, y_scale, y_min, plot_bottom, pad_l)}" '
                    f'class="line-pred dim-line dim-d{d}" data-dim="{d}" stroke="{color}"/>'
                )

            if ar_p is not None:
                ar_y = [row[d] for row in ar_p]
                paths += (
                    f'<path d="{_svg_path_str(list(range(T)), ar_y, x_scale, y_scale, y_min, plot_bottom, pad_l)}" '
                    f'class="line-ar dim-line dim-d{d}" data-dim="{d}" stroke="{color}"/>'
                )

        svg_content = (
            f'<svg viewBox="0 0 {svg_w} {svg_h}" class="action-chart" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<defs><clipPath id="clip-{uid}-{pidx}">'
            f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}"/>'
            f"</clipPath></defs>"
            f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>'
            f"{grid_lines}"
            f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{plot_bottom}" class="axis-line"/>'
            f'<line x1="{pad_l}" y1="{plot_bottom}" x2="{pad_l + plot_w}" y2="{plot_bottom}" class="axis-line"/>'
            f"{tick_labels}"
            f"{x_tick_labels}"
            f'<g clip-path="url(#clip-{uid}-{pidx})">{paths}</g>'
            f'<text x="{pad_l + plot_w / 2}" y="{svg_h - 2}" class="axis-title" text-anchor="middle">Time step</text>'
            f"</svg>"
        )
        dim_controls, dim_css = _build_dim_focus_controls(
            uid, pidx, dim, pred_p is not None, ar_p is not None
        )

        checked = " checked" if pidx == 0 else ""
        radio_inputs.append(
            f'<input type="radio" name="part-{uid}" id="part-{uid}-{pidx}" '
            f'class="part-radio"{checked}/>'
        )
        tab_labels.append(
            f'<label for="part-{uid}-{pidx}" class="part-tab">{pname} ({dim}D)</label>'
        )
        svg_panels.append(
            f'<div class="part-panel" id="panel-{uid}-{pidx}">'
            f"<style>{dim_css}</style>"
            f'<div class="dim-focus">{dim_controls}<div class="chart-wrap">{svg_content}</div></div>'
            f"</div>"
        )

    if not svg_panels:
        return ""

    label_html = f'<div class="section-label">{section_label}</div>' if section_label else ""

    tabs_css = ""
    for pidx in range(len(svg_panels)):
        selector = f"#part-{uid}-{pidx}:checked"
        tabs_css += f"{selector} ~ .part-tabs .part-tab:nth-child({pidx + 1}),"
        tabs_css += f"{selector}:checked ~ .part-tabs .part-tab:nth-child({pidx + 1}){{"
        tabs_css += "background:#4A90D9;color:#fff;}"
        tabs_css += f"{selector} ~ .part-panels #panel-{uid}-{pidx}{{display:block;}}"
    tabs_css = tabs_css.rstrip(",")

    parts_html = "\n".join(radio_inputs)
    tabs_html = '<div class="part-tabs">' + "\n".join(tab_labels) + "</div>"
    panels_html = '<div class="part-panels">' + "\n".join(svg_panels) + "</div>"

    style_id = f"style-{uid}"
    style_block = f'<style id="{style_id}">{tabs_css}</style>'

    return f'{label_html}<div class="action-plot">{style_block}{parts_html}{tabs_html}{panels_html}</div>'


def _nice_ticks(v_min, v_max, n_ticks):
    if not (math.isfinite(v_min) and math.isfinite(v_max)):
        return [0]
    rng = v_max - v_min
    if rng <= 0:
        return [v_min]
    rough = rng / n_ticks
    mag = 10 ** math.floor(math.log10(rough))
    res = rough / mag
    if res <= 1.5:
        step = 1 * mag
    elif res <= 3:
        step = 2 * mag
    elif res <= 7:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.ceil(v_min / step) * step
    ticks = []
    v = start
    while v <= v_max + step * 0.01:
        ticks.append(v)
        v += step
        if len(ticks) > n_ticks + 2:
            break
    return ticks


_CONTAINER_COUNTER = [0]


def _build_action_comparison_html(
    action_gt_denorm,
    action_pred_denorm,
    ar_pred_denorm,
    ar_gt_denorm,
    valid_dim_mask,
    ar_valid_dim_mask,
    parts_meta,
    action_gt_norm=None,
    action_pred_norm=None,
    ar_pred_norm=None,
    ar_gt_norm=None,
):
    has_fm = action_pred_denorm is not None
    has_ar = ar_pred_denorm is not None

    views = []

    if action_gt_denorm is not None and has_fm:
        html = _build_action_plot_html(
            action_gt_denorm,
            action_pred_denorm,
            None,
            parts_meta,
            section_label="FM vs GT (denormalized)",
        )
        if html:
            views.append(("fm-gt-denorm", "FM vs GT (denorm)", html))

    if action_gt_norm is not None and action_pred_norm is not None:
        html = _build_action_plot_html(
            action_gt_norm,
            action_pred_norm,
            None,
            parts_meta,
            section_label="FM vs GT (normalized)",
        )
        if html:
            views.append(("fm-gt-norm", "FM vs GT (norm)", html))

    if ar_gt_denorm is not None and has_ar:
        html = _build_action_plot_html(
            ar_gt_denorm,
            None,
            ar_pred_denorm,
            parts_meta,
            section_label="AR vs GT (denormalized)",
        )
        if html:
            views.append(("ar-gt-denorm", "AR vs GT (denorm)", html))

    if ar_gt_norm is not None and ar_pred_norm is not None:
        html = _build_action_plot_html(
            ar_gt_norm,
            None,
            ar_pred_norm,
            parts_meta,
            section_label="AR vs GT (normalized)",
        )
        if html:
            views.append(("ar-gt-norm", "AR vs GT (norm)", html))

    if action_gt_denorm is not None and has_fm and has_ar:
        html = _build_action_plot_html(
            action_gt_denorm,
            action_pred_denorm,
            ar_pred_denorm,
            parts_meta,
            section_label="All lines (denormalized)",
        )
        if html:
            views.append(("all-denorm", "All lines (denorm)", html))

    if action_gt_norm is not None and action_pred_norm is not None and ar_pred_norm is not None:
        html = _build_action_plot_html(
            action_gt_norm,
            action_pred_norm,
            ar_pred_norm,
            parts_meta,
            section_label="All lines (normalized)",
        )
        if html:
            views.append(("all-norm", "All lines (norm)", html))

    if not views:
        return ""

    _CONTAINER_COUNTER[0] += 1
    cuid = f"cc{_CONTAINER_COUNTER[0]}"

    radio_inputs = []
    view_labels = []
    view_panels = []
    view_css = []
    for i, (val, label, html) in enumerate(views):
        input_id = f"view-{cuid}-{i}"
        checked = " checked" if i == 0 else ""
        radio_inputs.append(
            f'<input type="radio" name="view-{cuid}" id="{input_id}" '
            f'class="chart-view-radio"{checked}/>'
        )
        view_labels.append(f'<label for="{input_id}" class="view-tab view-tab-{i}">{label}</label>')
        view_panels.append(
            f'<div class="chart-view-panel" id="view-panel-{cuid}-{i}" '
            f'data-view="{val}">{html}</div>'
        )
        view_css.append(
            f"#{input_id}:checked ~ .view-tabs .view-tab-{i}"
            "{background:#4A90D9;color:#fff;}"
            f"#{input_id}:checked ~ .view-panels #view-panel-{cuid}-{i}"
            "{display:block;}"
        )

    return (
        f'<div class="chart-container" id="chart-{cuid}">'
        f"<style>{''.join(view_css)}</style>"
        f"{''.join(radio_inputs)}"
        f'<div class="view-tabs">{"".join(view_labels)}</div>'
        f'<div class="view-panels">{"".join(view_panels)}</div>'
        f"</div>"
    )


_CSS = """
<style>
:root {
  --bg: #1a1b26; --surface: #24283b; --surface2: #2f3350; --border: #3b3f5c;
  --text: #c0caf5; --text2: #a9b1d6; --text3: #787c99; --accent: #7aa2f7;
  --accent2: #bb9af7; --green: #9ece6a; --red: #f7768e; --orange: #ff9e64;
  --yellow: #e0af68;
}
* { box-sizing: border-box; }
body { font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
       margin: 0; padding: 24px 32px; background: var(--bg); color: var(--text); line-height: 1.5; }
h1 { color: var(--accent); font-size: 1.3em; font-weight: 600; margin: 0 0 20px 0;
     padding-bottom: 10px; border-bottom: 1px solid var(--border); letter-spacing: 0.5px; }
h1 .step-badge { background: var(--accent); color: var(--bg); padding: 2px 8px;
                  border-radius: 4px; font-size: 0.85em; margin-left: 8px; }
.meta-row { color: var(--text3); font-size: 0.85em; margin-bottom: 20px; }

.sample-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
               padding: 20px; margin: 16px 0; }
.sample-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.sample-index { font-size: 1em; font-weight: 600; color: var(--accent); }
.meta-tag { display: inline-block; background: var(--surface2); color: var(--text2);
            padding: 2px 10px; border-radius: 4px; margin-left: 6px; font-size: 0.8em;
            border: 1px solid var(--border); }
.task-text { color: var(--text2); font-size: 0.88em; margin: 6px 0 12px 0; padding-left: 4px;
             border-left: 3px solid var(--accent2); }

.images-row { display: flex; gap: 10px; margin: 8px 0 16px 0; flex-wrap: wrap; }
.cam-cell { text-align: center; }
.cam-label { font-size: 0.7em; color: var(--text3); margin-bottom: 3px; }
.images-row img { border: 1px solid var(--border); border-radius: 6px; max-height: 180px;
                  background: #000; }

.proprio-block { background: var(--surface2); border-radius: 6px; padding: 8px 12px;
                 margin: 8px 0 16px 0; font-size: 0.78em; color: var(--text2);
                 border: 1px solid var(--border); }
.proprio-block .prop-label { color: var(--text3); font-size: 0.85em; margin-right: 6px; }

.action-plot { margin: 12px 0; }
.section-label { color: var(--accent2); font-size: 0.9em; font-weight: 600; margin: 16px 0 8px 0;
                 padding: 4px 0; border-bottom: 1px solid var(--border); }

.part-radio { display: none; }
.part-tabs { display: flex; gap: 2px; margin-bottom: 0; flex-wrap: wrap; }
.part-tab { padding: 5px 14px; font-size: 0.78em; color: var(--text2); background: var(--surface2);
            border: 1px solid var(--border); border-bottom: none; border-radius: 6px 6px 0 0;
            cursor: pointer; transition: background 0.15s, color 0.15s; user-select: none;
            font-family: inherit; }
.part-tab:hover { background: #3d4260; color: var(--text); }
.part-panels { border: 1px solid var(--border); border-radius: 0 6px 6px 6px; background: var(--surface);
               padding: 0; overflow: hidden; }
.part-panel { display: none; padding: 8px; }

.action-chart { width: 100%; height: auto; display: block; max-height: 340px; }
.plot-bg { fill: #1e2035; rx: 2; }
.grid-line { stroke: var(--border); stroke-width: 0.5; stroke-dasharray: 3,3; }
.axis-line { stroke: var(--text3); stroke-width: 1; }
.tick-label { fill: var(--text3); font-size: 9px; font-family: inherit; }
.axis-title { fill: var(--text3); font-size: 10px; font-family: inherit; }
.line-gt { fill: none; stroke-width: 1.8; }
.line-pred { fill: none; stroke-width: 1.5; stroke-dasharray: 7,3; }
.line-ar { fill: none; stroke-width: 1.2; stroke-dasharray: 2,2; opacity: 0.75; }

.dim-line { transition: opacity 0.2s ease; }
.dim-focus-radio { display: none; }
.dim-tabs { display: flex; gap: 6px; flex-wrap: wrap; padding: 8px;
            border-bottom: 1px solid var(--border); background: var(--surface2); }
.dim-focus-label { display: inline-flex; align-items: center; gap: 5px; padding: 3px 8px;
                   border: 1px solid var(--border); border-radius: 4px; color: var(--text2);
                   background: var(--surface); cursor: pointer; font-size: 0.72em;
                   user-select: none; transition: opacity 0.15s, background 0.15s, color 0.15s; }
.dim-focus-label:hover { border-color: var(--accent); color: var(--text); }
.legend-swatch { display: inline-block; width: 15px; height: 0; border-top-width: 2px;
                 border-top-style: solid; }
.swatch-pred { border-top-style: dashed; }
.swatch-ar { border-top-style: dotted; }
.chart-wrap { padding: 8px; }

.chart-container { position: relative; margin: 12px 0; }
.chart-view-radio { display: none; }
.view-tabs { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 8px; }
.view-tab { padding: 5px 12px; font-size: 0.78em; color: var(--text2);
            background: var(--surface2); border: 1px solid var(--border);
            border-radius: 4px; cursor: pointer; user-select: none; }
.view-tab:hover { border-color: var(--accent); color: var(--text); }
.chart-view-panel { display: none; }
</style>
"""

_JS = """
<script>
(function() {
  function hasClass(el, cls) {
    return el && (' ' + el.className + ' ').indexOf(' ' + cls + ' ') >= 0;
  }

  function findParentWithClass(el, cls) {
    while (el && el.nodeType === 1) {
      if (hasClass(el, cls)) return el;
      el = el.parentNode;
    }
    return null;
  }

  document.addEventListener('click', function(e) {
    e = e || window.event;
    var target = e.target || e.srcElement;
    var label = findParentWithClass(target, 'dim-focus-label');
    if (!label) return;

    var inputId = label.getAttribute('data-dim-input');
    var allId = label.getAttribute('data-all-input');
    if (!inputId || !allId) return;

    var input = document.getElementById(inputId);
    var allInput = document.getElementById(allId);
    if (input && allInput && input.checked) {
      allInput.checked = true;
      if (e.preventDefault) e.preventDefault();
      e.returnValue = false;
    }
  });
})();
</script>
"""


def save_eval_snapshot(
    batch,
    preds_out,
    output_dir,
    step,
    parts_meta=None,
    camera_names=None,
    max_samples=None,
):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    try:
        _save_eval_snapshot_impl(
            batch, preds_out, output_dir, step, parts_meta, camera_names, max_samples
        )
    except Exception:
        logger.warning("Failed to save eval snapshot at step %s, skipping", step, exc_info=True)


def _save_eval_snapshot_impl(
    batch, preds_out, output_dir, step, parts_meta, camera_names, max_samples
):
    snapshot_dir = Path(output_dir) / "eval_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    html_path = snapshot_dir / f"step_{step}.html"

    sample_meta_list = batch.get("sample_meta", [])
    B = _get_batch_size(batch)
    if max_samples is not None:
        B = min(B, max_samples)

    action_gt_denorm = preds_out.get("action_gt_denorm") if preds_out else None
    action_pred_denorm = preds_out.get("action_denorm") if preds_out else None
    ar_pred_denorm = preds_out.get("ar_action_denorm") if preds_out else None
    ar_gt_denorm = preds_out.get("ar_gt_denorm") if preds_out else None
    valid_dim_mask = preds_out.get("valid_dim_mask") if preds_out else None
    ar_valid_dim_mask = preds_out.get("ar_valid_dim_mask") if preds_out else None
    action_gt_norm = preds_out.get("action_gt") if preds_out else None
    action_pred_norm = preds_out.get("action") if preds_out else None
    ar_pred_norm = preds_out.get("ar_action_norm") if preds_out else None
    ar_gt_norm = preds_out.get("ar_gt_norm") if preds_out else None

    samples_html = []
    for i in range(B):
        meta = sample_meta_list[i] if i < len(sample_meta_list) else {}
        task = meta.get("task", "N/A")
        embodiment = meta.get("embodiment", "N/A")
        ds_locator = meta.get("dataset_locator", "N/A")

        images_html = _render_images_for_sample(batch, i, camera_names)

        gt_i = action_gt_denorm[i] if action_gt_denorm is not None else None
        pred_i = action_pred_denorm[i] if action_pred_denorm is not None else None
        ar_pred_i = ar_pred_denorm[i] if ar_pred_denorm is not None else None
        ar_gt_i = ar_gt_denorm[i] if ar_gt_denorm is not None else None
        vmask_i = valid_dim_mask[i] if valid_dim_mask is not None else None
        ar_vmask_i = ar_valid_dim_mask[i] if ar_valid_dim_mask is not None else None
        gt_norm_i = action_gt_norm[i] if action_gt_norm is not None else None
        pred_norm_i = action_pred_norm[i] if action_pred_norm is not None else None
        ar_pred_norm_i = ar_pred_norm[i] if ar_pred_norm is not None else None
        ar_gt_norm_i = ar_gt_norm[i] if ar_gt_norm is not None else None

        action_html = _build_action_comparison_html(
            gt_i,
            pred_i,
            ar_pred_i,
            ar_gt_i,
            vmask_i,
            ar_vmask_i,
            parts_meta,
            action_gt_norm=gt_norm_i,
            action_pred_norm=pred_norm_i,
            ar_pred_norm=ar_pred_norm_i,
            ar_gt_norm=ar_gt_norm_i,
        )

        proprio = batch.get("proprio")
        proprio_html = ""
        if proprio is not None:
            proprio_val = proprio[i].cpu() if isinstance(proprio, torch.Tensor) else proprio[i]
            if isinstance(proprio_val, torch.Tensor):
                proprio_val = proprio_val.tolist()
            if proprio_val:
                proprio_html = (
                    f'<div class="proprio-block"><span class="prop-label">Proprio</span>'
                    f"{_fmt_vec(proprio_val[-1] if proprio_val else [])}</div>"
                )

        samples_html.append(
            f"""<div class="sample-card">
  <div class="sample-header">
    <span class="sample-index">Sample {i}</span>
    <span><span class="meta-tag">{embodiment}</span>
    <span class="meta-tag">{ds_locator}</span></span>
  </div>
  <div class="task-text">{task}</div>
  {images_html}
  {proprio_html}
  {action_html}
</div>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Eval Snapshot — Step {step}</title>
{_CSS}
</head><body>
<h1>Eval Snapshot<span class="step-badge">Step {step}</span></h1>
<div class="meta-row">Batch size: {B} &nbsp;|&nbsp; Parts: {parts_meta or "N/A"}</div>
{"".join(samples_html)}
{_JS}
</body></html>"""

    html_path.write_text(html, encoding="utf-8")
    logger.info("Saved eval snapshot to %s", html_path)


def save_train_snapshot(batch, output_dir, parts_meta=None, camera_names=None):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    try:
        _save_train_snapshot_impl(batch, output_dir, parts_meta, camera_names)
    except Exception:
        logger.warning("Failed to save train snapshot, skipping", exc_info=True)


def _save_train_snapshot_impl(batch, output_dir, parts_meta, camera_names):
    snapshot_dir = Path(output_dir) / "train_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    html_path = snapshot_dir / "first_batch.html"

    sample_meta_list = batch.get("sample_meta", [])
    B = _get_batch_size(batch)

    action = batch.get("action")

    samples_html = []
    for i in range(B):
        meta = sample_meta_list[i] if i < len(sample_meta_list) else {}
        task = meta.get("task", "N/A")
        embodiment = meta.get("embodiment", "N/A")
        ds_locator = meta.get("dataset_locator", "N/A")

        images_html = _render_images_for_sample(batch, i, camera_names)

        action_html = ""
        if action is not None:
            action_val = action[i].cpu() if isinstance(action, torch.Tensor) else action[i]
            if isinstance(action_val, torch.Tensor):
                action_val = action_val.tolist()
            action_html = _build_action_plot_html(
                action_val,
                None,
                None,
                parts_meta,
                section_label="GT Action (normalized)",
            )

        proprio = batch.get("proprio")
        proprio_html = ""
        if proprio is not None:
            proprio_val = proprio[i].cpu() if isinstance(proprio, torch.Tensor) else proprio[i]
            if isinstance(proprio_val, torch.Tensor):
                proprio_val = proprio_val.tolist()
            if proprio_val:
                proprio_html = (
                    f'<div class="proprio-block"><span class="prop-label">Proprio</span>'
                    f"{_fmt_vec(proprio_val[-1] if proprio_val else [])}</div>"
                )

        samples_html.append(
            f"""<div class="sample-card">
  <div class="sample-header">
    <span class="sample-index">Sample {i}</span>
    <span><span class="meta-tag">{embodiment}</span>
    <span class="meta-tag">{ds_locator}</span></span>
  </div>
  <div class="task-text">{task}</div>
  {images_html}
  {proprio_html}
  {action_html}
</div>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Train Snapshot — First Batch</title>
{_CSS}
</head><body>
<h1>Train Snapshot<span class="step-badge">First Batch</span></h1>
<div class="meta-row">Batch size: {B} &nbsp;|&nbsp; Parts: {parts_meta or "N/A"}</div>
{"".join(samples_html)}
{_JS}
</body></html>"""

    html_path.write_text(html, encoding="utf-8")
    logger.info("Saved train snapshot to %s", html_path)
