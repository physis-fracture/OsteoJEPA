"""Render the parts of the architecture figure that must not be invented.

A diagram generator will draw a chart that looks like a chart. It will not draw
*this* chart, and the first attempt produced one sloping the wrong way, which
argued the opposite of the finding. Two elements carry claims rather than
decoration, so both are rendered here from the artifacts and handed over as
finished SVG for the figure to embed:

* the skill-ratio curves, from `artifacts/probe/probe_{a,b,c}.json`
* a real radiograph with its patch grid and valid mask, from the test fold

Everything else in the figure is layout, which a generator is good at.

    python scripts/make_figure_assets.py --out docs/figures
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.geometry import valid_mask_from_geometry
from physis.utils.config import load_config

PROBES = [
    ("probe_a", "baseline", "#1A1A1A"),
    ("probe_b", "variance weight 25 → 1", "#7A6BA8"),
    ("probe_c", "border erosion, 1 ring", "#5B8C6E"),
]
ACCENT = "#C0392B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="render figure assets from artifacts")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--probes", default="artifacts/probe")
    parser.add_argument("--stem", default="1519_0859295589_05_WRI-R1_M011")
    parser.add_argument("--delta-map", default="runs/e1/figures/delta_spatial.npy")
    parser.add_argument(
        "--layout",
        default=None,
        help="a generated layout SVG; substitutes its placeholders and writes figure.svg",
    )
    return parser.parse_args()


def skill_chart(probe_dir: Path) -> str:
    """The three probe curves, drawn from the epoch-level metrics.

    Epoch 0 is dropped from the plotted range and called out separately. It sits
    at ~1.36 because the FiLM MLP is zero-initialized, so the predictor starts
    worse than a constant and the first epoch is not evidence of anything.
    """
    series = []
    for name, label, colour in PROBES:
        rows = json.loads((probe_dir / f"{name}.json").read_text(encoding="utf-8"))
        series.append((label, colour, [(r["epoch"], r["pred_skill_ratio"]) for r in rows]))

    W, H = 520, 300
    L, R, T, B = 62, 150, 42, 44
    y_lo, y_hi = 0.970, 1.010
    x_lo, x_hi = 1, max(e for _, _, s in series for e, _ in s)

    def px(e):
        return L + (e - x_lo) / (x_hi - x_lo) * (W - L - R)

    def py(v):
        return T + (y_hi - v) / (y_hi - y_lo) * (H - T - B)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Inter, Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
        f'<text x="{L}" y="20" font-size="12" font-weight="600" fill="#1A1A1A">'
        f'Prediction skill against a constant predictor</text>',
        f'<text x="{L}" y="34" font-size="9" fill="#666">'
        f'L_pred divided by the target tokens\' own variance. Below 1.0 is better than the mean.</text>',
    ]

    # Gridlines and the y axis. Ticks every 0.01, which is the scale the whole
    # effect lives on: the full vertical range here is 4%.
    for v in np.arange(0.97, 1.0101, 0.01):
        y = py(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#EDEDED"/>')
        out.append(
            f'<text x="{L-8}" y="{y+3:.1f}" font-size="9" fill="#666" '
            f'text-anchor="end">{v:.2f}</text>'
        )

    # The line that carries the argument.
    y1 = py(1.0)
    out.append(
        f'<line x1="{L}" y1="{y1:.1f}" x2="{W-R}" y2="{y1:.1f}" '
        f'stroke="{ACCENT}" stroke-width="1.5" stroke-dasharray="5 3"/>'
    )
    out.append(
        f'<text x="{W-R+6}" y="{y1+3:.1f}" font-size="9" fill="{ACCENT}">'
        f'1.0 = no skill at all</text>'
    )

    for k, (label, colour, points) in enumerate(series):
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{px(e):.1f},{py(min(max(v, y_lo), y_hi)):.1f}"
            for i, (e, v) in enumerate(p for p in points if p[0] >= x_lo)
        )
        out.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.6"/>')
        last = points[-1][1]
        out.append(
            f'<text x="{W-R+6}" y="{py(last)+3+ (k-1)*11:.1f}" font-size="9" '
            f'fill="{colour}">{label} → {last:.3f}</text>'
        )

    out.append(
        f'<text x="{L}" y="{H-24}" font-size="9" fill="#666">epoch 1 '
        f'<tspan x="{W-R-12}" text-anchor="end">epoch {x_hi}</tspan></text>'
    )
    # The shape of the result, stated rather than left to the reader.
    out.append(
        f'<text x="{L}" y="{H-8}" font-size="9.5" fill="#1A1A1A">'
        f'Best skill arrives at epoch 2 and is lost again: training made prediction worse.</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def patch_grid(cfg, stem: str) -> str:
    """A real radiograph, its 24 x 24 patch grid, and the valid mask."""
    from PIL import Image
    import pandas as pd

    manifest = pd.read_csv(cfg.manifest)
    row = manifest[manifest["stem"] == stem].iloc[0]
    array = np.asarray(Image.open(Path(cfg.images_dir) / f"{stem}.png"), dtype=np.float32)
    mask = valid_mask_from_geometry(
        float(row.pad_x), float(row.pad_y), float(row.new_w), float(row.new_h),
        size=int(cfg.image.size), patch=int(cfg.image.patch),
    )

    eight = (array / max(array.max(), 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(eight).save(buffer, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    S = 288                       # rendered size of the canvas
    cell = S / mask.shape[0]
    valid = int(mask.sum())
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S+34}" '
        f'viewBox="0 0 {S} {S+34}" font-family="Inter, Helvetica, Arial, sans-serif">',
        f'<image href="{data_uri}" x="0" y="0" width="{S}" height="{S}"/>',
    ]
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            x, y = i * cell, j * cell
            if mask[j, i]:
                out.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell:.2f}" height="{cell:.2f}" '
                    f'fill="none" stroke="#FFFFFF" stroke-opacity="0.30" stroke-width="0.4"/>'
                )
            else:
                out.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell:.2f}" height="{cell:.2f}" '
                    f'fill="{ACCENT}" fill-opacity="0.30" stroke="{ACCENT}" '
                    f'stroke-opacity="0.45" stroke-width="0.4"/>'
                )
    out.append(
        f'<text x="0" y="{S+14}" font-size="9.5" fill="#1A1A1A">'
        f'24 x 24 patches of 16 px = 576 tokens</text>'
    )
    out.append(
        f'<text x="0" y="{S+28}" font-size="9.5" fill="{ACCENT}">'
        f'{576 - valid} padding tokens excluded everywhere ({(576-valid)/576*100:.0f}%)</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def delta_map(path: Path, cfg, stem: str) -> str:
    """The mean Delta map laid over a radiograph. One image, no chart.

    An earlier version put the two side by side with a column profile beneath
    and a set of annotations. It was correct and it was unreadable: too wide to
    fit, and the eye had to travel between three panels to make one comparison.
    Overlaid, the comparison is immediate. The heat sits on the outline of the
    limb and not on the bones.

    The radiograph stays pure greyscale at full opacity and the heat is alpha
    over it. Tinting both red, as the first attempt did, made them the same hue
    and neither could be read.
    """
    from PIL import Image

    grid = np.load(path).astype(np.float64)
    n = grid.shape[0]
    columns = grid.mean(axis=0)
    lo, hi = float(grid.min()), float(grid.max())
    centre = float(grid[8:16, 8:16].mean())
    left = int(np.argmax(columns[:8]))
    right = int(np.argmax(columns[16:])) + 16

    array = np.asarray(Image.open(Path(cfg.images_dir) / f"{stem}.png"), dtype=np.float32)
    eight = (array / max(array.max(), 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(eight).save(buffer, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    S = 460
    cell = S / n
    T = 46
    W, H = S, T + S + 44

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Inter, Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
        f'<text x="0" y="18" font-size="13.5" font-weight="600" fill="#1A1A1A">'
        f'Measured age-attribution map</text>',
        f'<text x="0" y="34" font-size="10" fill="#666">'
        f'Mean Delta per patch over clean images, on a study of the same canvas.</text>',
        f'<image href="{data_uri}" x="0" y="{T}" width="{S}" height="{S}"/>',
    ]

    # Alpha only, over an untinted radiograph. Squaring the normalized value
    # holds the mid-range back so the two peaks stay distinguishable from the
    # broad low plateau rather than washing into it.
    for j in range(n):
        for i in range(n):
            t = (grid[j, i] - lo) / max(hi - lo, 1e-12)
            out.append(
                f'<rect x="{i*cell:.2f}" y="{T+j*cell:.2f}" width="{cell:.2f}" '
                f'height="{cell:.2f}" fill="{ACCENT}" fill-opacity="{t*t*0.88:.3f}"/>'
            )
    out.append(f'<rect x="0" y="{T}" width="{S}" height="{S}" fill="none" stroke="#9AA0A6"/>')

    out.append(
        f'<text x="0" y="{T+S+18}" font-size="10.5" font-weight="600" fill="#1A1A1A">'
        f'Delta peaks on the edges of the limb ({columns[left]:.4f}, {columns[right]:.4f}), '
        f'not on the bones ({centre:.4f}).</text>'
    )
    out.append(
        f'<text x="0" y="{T+S+34}" font-size="9.5" fill="#666">'
        f'Limb width in the frame is body size, and body size is age.</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def compose(layout: Path, assets: dict[str, Path], out: Path) -> None:
    """Drop the rendered assets into the layout's placeholder rectangles.

    The layout comes from a diagram generator, which cannot be handed a 141 KB
    SVG carrying a base64 radiograph. It leaves a `<rect id="asset-skill">` and a
    `<rect id="asset-grid">` instead, and this substitutes them, scaling each
    asset to the rectangle it was given.

    A placeholder that is missing is reported rather than skipped: a figure
    silently short one panel is worse than one that refuses to build.
    """
    import re

    svg = layout.read_text(encoding="utf-8")
    for placeholder_id, path in assets.items():
        match = re.search(
            rf'<rect[^>]*id="{placeholder_id}"[^>]*/>', svg
        ) or re.search(rf'<rect[^>]*id="{placeholder_id}"[^>]*>.*?</rect>', svg, re.S)
        if match is None:
            raise SystemExit(
                f'no placeholder id="{placeholder_id}" in {layout}\n'
                "The generator must leave one empty <rect> per asset, with that id."
            )
        tag = match.group(0)
        box = {k: float(v) for k, v in re.findall(r'\b(x|y|width|height)="([-\d.]+)"', tag)}

        asset = path.read_text(encoding="utf-8")
        size = re.search(r'width="([\d.]+)"\s+height="([\d.]+)"', asset)
        aw, ah = float(size.group(1)), float(size.group(2))
        scale = min(box["width"] / aw, box["height"] / ah)

        inner = asset[asset.index(">", asset.index("<svg")) + 1 : asset.rindex("</svg>")]
        svg = svg.replace(
            tag,
            f'<g transform="translate({box["x"]:.1f},{box["y"]:.1f}) '
            f'scale({scale:.4f})">{inner}</g>',
        )
        print(f"  placed {path.name} at {box['x']:.0f},{box['y']:.0f} scale {scale:.3f}")

    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.layout:
        compose(
            Path(args.layout),
            {
                "asset-skill": out / "skill_ratio.svg",
                "asset-grid": out / "patch_grid.svg",
                "asset-delta": out / "delta_map.svg",
            },
            out / "figure.svg",
        )
        return

    cfg = load_config(args.config)
    (out / "skill_ratio.svg").write_text(skill_chart(Path(args.probes)), encoding="utf-8")
    print(f"wrote {out / 'skill_ratio.svg'}")

    (out / "patch_grid.svg").write_text(patch_grid(cfg, args.stem), encoding="utf-8")
    print(f"wrote {out / 'patch_grid.svg'}")

    source = Path(args.delta_map)
    if source.exists():
        (out / "delta_map.svg").write_text(delta_map(source, cfg, args.stem), encoding="utf-8")
        print(f"wrote {out / 'delta_map.svg'}")
    else:
        print(f"skipped delta_map.svg: {source} not found (pull runs/e1 from Modal)")
    print("\nNext: paste the prompt, save the layout SVG, then")
    print(f"  python scripts/make_figure_assets.py --layout <layout.svg> --out {out}")


if __name__ == "__main__":
    main()
