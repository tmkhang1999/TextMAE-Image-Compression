"""
Regenerate the README figures from the real scoring / selection code.

    python tools/make_figures.py

Writes to assets/figures/:
    concept.svg           standard learned codec vs. TextMAE
    pipeline.svg          method figure (thumbnails embedded)
    patch_selection.png   score maps and the kept patches (Kodak kodim23)
    rd_kodak.png          rate-distortion points of TextMAE vs. JPEG / WebP
    refinement.jpg        decoded vs. caption-guided refinement (needs tools/refine_example.py)

TextMAE numbers and the decoded example come from assets/1.png and assets/2.png
(outputs of the trained model of the original experiments). The refined example and
its caption come from tools/refine_example.py. JPEG / WebP are computed here with
Pillow on the same 224 x 224 inputs.
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textmae.data.patch_scores import image_patch_scores  # noqa: E402
from textmae.data.score_maps import structure_map, texture_map  # noqa: E402
from textmae.models.patch_selection import select_patches  # noqa: E402

INPUT_SIZE, PATCH, KEEP = 224, 16, 144
GRID = INPUT_SIZE // PATCH
DROP_GRAY = (138, 136, 126)  # same gray as the masked inputs in assets/1.png
BLUE, TEAL, GRAY, INK = "#2B6CB0", "#2C7A7B", "#8A8A8A", "#333333"

# (bpp, PSNR dB) read from the labels of assets/1.png (kodim20) and assets/2.png (kodim23)
TEXTMAE_POINTS = {
    "kodim20": [(0.020, 22.44), (0.07, 25.6), (0.15, 27.8)],
    "kodim23": [(0.018, 22.2), (0.06, 26.1), (0.12, 27.5)],
}


def kept_mask(scores, strategy, seed=0):
    """Boolean (GRID, GRID) mask of the patches kept by `strategy`."""
    torch.manual_seed(seed)
    ids = select_patches(strategy, torch.from_numpy(scores)[None], KEEP)[0, :KEEP]
    mask = np.zeros(GRID * GRID, dtype=bool)
    mask[ids.numpy()] = True
    return mask.reshape(GRID, GRID)


def masked_view(rgb, mask):
    """Paint dropped patches gray and draw a faint patch grid."""
    out = rgb.copy()
    big = np.kron(mask, np.ones((PATCH, PATCH), dtype=bool))
    out[~big] = DROP_GRAY
    for k in range(1, GRID):
        out[k * PATCH, :] = (out[k * PATCH, :] * 0.75).astype(np.uint8)
        out[:, k * PATCH] = (out[:, k * PATCH] * 0.75).astype(np.uint8)
    return out


def score_heatmap(scores):
    """Patch scores as an RGB image (magma colormap), upsampled with hard patch edges."""
    rgba = plt.get_cmap("magma")(scores.reshape(GRID, GRID))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return np.kron(rgb, np.ones((PATCH, PATCH, 1), dtype=np.uint8))


def load_inputs(image_path):
    rgb = np.array(Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC))
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"Could not read {image_path}")

    size = (INPUT_SIZE, INPUT_SIZE)
    s_map = structure_map(gray.copy(), size)
    t_map = texture_map(gray, size)
    scores = image_patch_scores(gray, INPUT_SIZE, PATCH)
    return rgb, s_map, t_map, scores


def make_patch_figure(rgb, s_map, t_map, scores, out_path):
    panels = [
        (rgb, None, "(a) Input\n224 x 224, 14 x 14 patches"),
        (s_map, "gray", "(b) Structure map\nsplit-and-merge segmentation"),
        (t_map, "gray", "(c) Texture map\n|Laplacian|, white = edges"),
        (scores.reshape(GRID, GRID), "magma", "(d) Patch score\ntexture x structure"),
        (masked_view(rgb, kept_mask(scores, "stratified")), None,
         f"(e) Kept, percentile sampling\n{KEEP} of {GRID * GRID} patches (default)"),
        (masked_view(rgb, kept_mask(scores, "multinomial")), None,
         f"(f) Kept, multinomial sampling\n{KEEP} of {GRID * GRID} patches"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 3.4), dpi=160)
    for ax, (img, cmap, title) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#c8c8c8")
        ax.set_xlabel(title, fontsize=10.5, linespacing=1.35, labelpad=6, color=INK)
        if cmap == "magma":
            # Colorbar in an inset so this panel keeps the same size as the others
            cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
            cbar = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
            cbar.ax.tick_params(labelsize=8, colors=INK)
            cbar.outline.set_edgecolor("#c8c8c8")
    fig.patch.set_facecolor("white")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def codec_curve(image_path, fmt, qualities):
    """(bpp, PSNR) of Pillow JPEG / WebP on the 224 x 224 input, same protocol as evaluate.py."""
    img = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
    ref = np.asarray(img, dtype=np.float64)
    points = []
    for q in qualities:
        buf = io.BytesIO()
        options = {"quality": q, "optimize": True} if fmt == "JPEG" else {"quality": q, "method": 6}
        img.save(buf, fmt, **options)
        rec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float64)
        mse = np.mean((ref - rec) ** 2)
        points.append((buf.tell() * 8 / INPUT_SIZE ** 2, 10 * np.log10(255 ** 2 / mse)))
    return points


def make_rd_figure(out_path):
    titles = {"kodim20": "Kodak kodim20 (aircraft)", "kodim23": "Kodak kodim23 (parrots)"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160, sharey=True)
    for ax, (name, title) in zip(axes, titles.items()):
        path = ROOT / "datasets" / "kodak" / f"{name}.png"
        for fmt, color, marker, label in (("JPEG", GRAY, "s", "JPEG"), ("WEBP", TEAL, "^", "WebP")):
            pts = codec_curve(path, fmt, [1, 3, 5, 10, 15, 20, 30, 40])
            ax.plot(*zip(*pts), color=color, marker=marker, ms=4, lw=1.4, label=label)
        ax.plot(*zip(*TEXTMAE_POINTS[name]), color=BLUE, marker="o", ms=6, lw=2, label="TextMAE")
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xlabel("bits per pixel (lower = smaller file)", fontsize=10, color=INK)
        ax.set_xlim(0, 0.5)
        ax.grid(True, color="#e5e5e5", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].set_ylabel("PSNR in dB (higher = more faithful)", fontsize=10, color=INK)
    fig.text(0.5, -0.02,
             "All codecs on the same 224 x 224 inputs. JPEG / WebP: Pillow, measured with tools/make_figures.py. "
             "TextMAE: values labeled in the result figures of the original experiments.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# ---------------------------------------------------------------- example images

# Bottom-right tile of assets/2.png: the 0.12 bpp / 27.5 dB reconstruction of kodim23.
# Its top-right corner carries a burned-in "bpp / dB" label, which the figures cover
# with their own label box.
DECODED_TILE = (595, 300, 738, 443)
TILE_LABEL = (104 / 143, 0.0, 1.0, 33 / 143)  # label box as fractions of the tile


def decoded_rgb():
    return np.array(Image.open(ROOT / "assets" / "2.png").convert("RGB").crop(DECODED_TILE))


def refined_example():
    """Real SDXL output from tools/refine_example.py, or None if it was not generated."""
    img_path = ROOT / "assets" / "figures" / "kodim23_refined.png"
    meta_path = ROOT / "assets" / "figures" / "kodim23_caption.json"
    if not img_path.exists() or not meta_path.exists():
        print("No refined example yet (run tools/refine_example.py); using a placeholder")
        return None, None
    return np.array(Image.open(img_path).convert("RGB")), json.loads(meta_path.read_text())


def jpeg_data_uri(rgb, pixels=192, quality=88):
    """Thumbnails are embedded once per figure as JPEG to keep the SVG files small."""
    img = Image.fromarray(rgb).resize((pixels, pixels), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- SVG drawing helpers
# Figures follow the usual learned-compression paper conventions: trapezoids for
# encoders / decoders, a black-and-white strip for bitstreams, AE / AD for the
# arithmetic coder, math symbols on arrows and light panels for grouping.

FONT = "Arial, Helvetica, sans-serif"
MATH = "'Times New Roman', Times, serif"
PANEL_FILL = "#F3F5F8"
SIDE_COLOR = TEAL      # side information that is transmitted (patch positions, caption)
PROB_COLOR = "#8A8A8A"  # probabilities from the entropy model (not transmitted)

STYLE = """    <style>
      text  {{ fill: #333333; font-family: {font}; }}
      .lbl  {{ font-size: 13px; }}
      .bold {{ font-size: 13px; font-weight: 700; }}
      .head {{ font-size: 14px; font-weight: 700; }}
      .note {{ font-size: 12.5px; fill: #555555; }}
      .tag  {{ font-size: 9px; font-weight: 700; fill: #333333; }}
      .m    {{ font-family: {math}; font-style: italic; font-size: 17px; }}
      .ms   {{ font-family: {math}; font-style: italic; font-size: 12.5px; }}
      .flow {{ fill: none; stroke: #333333; stroke-width: 1.4; marker-end: url(#arr); }}
      .side {{ fill: none; stroke: {side}; stroke-width: 1.4; stroke-dasharray: 6 3; marker-end: url(#arr-side); }}
      .prob {{ fill: none; stroke: {prob}; stroke-width: 1.3; stroke-dasharray: 2 3; marker-end: url(#arr-prob); }}
      .enc  {{ fill: #CFE0F5; stroke: #333333; stroke-width: 1.2; }}
      .dec  {{ fill: #D5EBD9; stroke: #333333; stroke-width: 1.2; }}
      .box  {{ fill: #FFFFFF; stroke: #333333; stroke-width: 1.2; }}
      .thumb {{ fill: none; stroke: #333333; stroke-width: 1; }}
    </style>
""".format(font=FONT, math=MATH, side=SIDE_COLOR, prob=PROB_COLOR)


def _marker(marker_id, color):
    return (f'    <marker id="{marker_id}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" '
            f'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">'
            f'<path d="M0,1 L10,5 L0,9 z" fill="{color}"/></marker>\n')


class Svg:
    """Collects body elements and embeds every image once in <defs>."""

    def __init__(self, width, height, title, desc, top=0):
        # `top` crops empty space above the content without moving every coordinate
        self.width, self.height, self.top = width, height, top
        self.title, self.desc = title, desc
        self.images = {}
        self.body = []

    def add(self, element):
        self.body.append(element)

    def text(self, x, y, s, cls="lbl", anchor="middle"):
        self.add(f'  <text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>\n')

    def path(self, d, cls="flow"):
        self.add(f'  <path class="{cls}" d="{d}"/>\n')

    def image(self, x, y, size, key, rgb):
        if key not in self.images:
            self.images[key] = jpeg_data_uri(rgb)
        self.add(f'  <use xlink:href="#img-{key}" href="#img-{key}" '
                 f'transform="translate({x},{y}) scale({size})"/>\n'
                 f'  <rect class="thumb" x="{x}" y="{y}" width="{size}" height="{size}"/>\n')

    def tile_label(self, x, y, size, text):
        """Cover the burned-in label of a result tile with a readable one."""
        x0, y0, x1, y1 = TILE_LABEL
        self.add(f'  <rect x="{x + x0 * size:.1f}" y="{y + y0 * size:.1f}" width="{(x1 - x0) * size:.1f}" '
                 f'height="{(y1 - y0) * size:.1f}" fill="#FFFFFF" stroke="#333333" stroke-width="0.8"/>\n')
        self.add(f'  <text class="tag" x="{x + (x0 + x1) / 2 * size:.1f}" y="{y + (y0 + y1) / 2 * size + 3:.1f}" '
                 f'text-anchor="middle">{text}</text>\n')

    def placeholder(self, x, y, size, text):
        self.add(f'  <rect x="{x}" y="{y}" width="{size}" height="{size}" fill="#FFFFFF" '
                 f'stroke="#333333" stroke-dasharray="4 3"/>\n')
        self.text(x + size / 2, y + size / 2 + 4, text, "note")

    def trapezoid(self, x, y, w, h, shrink, narrow_right, cls):
        """Encoder (narrows to the right) or decoder (widens to the left)."""
        if narrow_right:
            pts = [(x, y), (x + w, y + shrink), (x + w, y + h - shrink), (x, y + h)]
        else:
            pts = [(x, y + shrink), (x + w, y), (x + w, y + h), (x, y + h - shrink)]
        self.add(f'  <polygon class="{cls}" points="{" ".join(f"{px},{py}" for px, py in pts)}"/>\n')

    def strip(self, x, y, cells, cell=10, thick=14, vertical=False):
        """Bitstream icon: a fixed black / white pattern of `cells` squares."""
        pattern = "1011001110100101110010110"
        for i in range(cells):
            fill = "#222222" if pattern[i % len(pattern)] == "1" else "#FFFFFF"
            cx, cy = (x, y + i * cell) if vertical else (x + i * cell, y)
            w, h = (thick, cell) if vertical else (cell, thick)
            self.add(f'  <rect x="{cx}" y="{cy}" width="{w}" height="{h}" fill="{fill}" '
                     f'stroke="#222222" stroke-width="0.8"/>\n')

    def snowflake(self, cx, cy, r=7):
        """'Frozen' marker for pretrained models used without training."""
        lines = []
        for k in range(3):
            a = np.pi * k / 3
            dx, dy = r * np.cos(a), r * np.sin(a)
            lines.append(f'<line x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" x2="{cx + dx:.1f}" y2="{cy + dy:.1f}"/>')
        self.add(f'  <g stroke="#2B6CB0" stroke-width="1.6" stroke-linecap="round">{"".join(lines)}</g>\n')

    def panel(self, x, y, w, h, title):
        self.add(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{PANEL_FILL}"/>\n')
        self.text(x + 14, y + 22, title, "head", "start")

    def write(self, out_path):
        images = "".join(
            f'    <image id="img-{key}" width="1" height="1" preserveAspectRatio="none" '
            f'xlink:href="{uri}" href="{uri}"/>\n' for key, uri in self.images.items())
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
               f'viewBox="0 {self.top} {self.width} {self.height - self.top}" width="{self.width}" '
               f'height="{self.height - self.top}" '
               f'role="img" aria-labelledby="title desc">\n'
               f'  <title id="title">{self.title}</title>\n  <desc id="desc">{self.desc}</desc>\n'
               f'  <defs>\n{_marker("arr", "#333333")}{_marker("arr-side", SIDE_COLOR)}'
               f'{_marker("arr-prob", PROB_COLOR)}{STYLE}{images}  </defs>\n'
               f'  <rect y="{self.top}" width="{self.width}" height="{self.height}" fill="#FFFFFF"/>\n'
               + "".join(self.body) + "</svg>\n")
        out_path.write_text(svg, encoding="ascii")


def _short(text, limit):
    text = text.replace("&", "and").replace("<", "").replace(">", "").replace('"', "'")
    if len(text) <= limit:
        return text
    return text[:limit - 3].rsplit(" ", 1)[0] + "..."  # cut at a word boundary


def make_concept_svg(ex, out_path):
    """'Standard learned codec vs TextMAE' at a glance."""
    svg = Svg(960, 372, "Standard learned codec vs TextMAE",
              "A standard learned codec encodes all 196 patches. TextMAE encodes only the 144 most "
              "informative patches plus a short caption; an MAE decoder fills in the dropped patches "
              "and a caption-guided SDXL refiner restores detail.")
    # Row 1: standard learned codec (output shown as an illustration)
    svg.text(24, 36, "Standard learned codec", "head", "start")
    svg.image(24, 52, 72, "input", ex["input"])
    svg.path("M96,88 H124")
    svg.trapezoid(124, 56, 48, 64, 12, True, "enc")
    svg.text(148, 92, "Enc", "bold")
    svg.path("M172,88 H196")
    svg.strip(196, 81, 14)
    svg.text(266, 116, "196 patches", "note")
    svg.path("M336,88 H360")
    svg.trapezoid(360, 56, 48, 64, 12, False, "dec")
    svg.text(384, 92, "Dec", "bold")
    svg.path("M408,88 H432")
    svg.image(432, 52, 72, "input", ex["input"])
    svg.text(468, 142, '<tspan class="m">x&#770;</tspan> <tspan class="note">(illustration)</tspan>')

    svg.add('  <line x1="16" y1="164" x2="944" y2="164" stroke="#BBBBBB" stroke-dasharray="6 4"/>\n')

    # Row 2: TextMAE (centre y = 284); the caption bypasses the codec and goes to the refiner
    svg.text(24, 196, "TextMAE (ours)", "head", "start")
    svg.image(24, 248, 72, "input", ex["input"])
    svg.path("M96,284 H124")
    svg.image(124, 248, 72, "kept", ex["kept"])
    svg.text(160, 340, "keep informative", "note")
    svg.text(160, 355, "patches", "note")
    svg.path("M196,284 H220")
    svg.trapezoid(220, 252, 48, 64, 12, True, "enc")
    svg.text(244, 288, "Enc", "bold")
    svg.path("M268,284 H292")
    svg.strip(292, 277, 10)
    svg.text(342, 340, "144 patches", "note")
    svg.path("M392,284 H416")
    svg.trapezoid(416, 252, 48, 64, 12, False, "dec")
    svg.text(440, 288, "Dec", "bold")
    svg.text(440, 340, "MAE fills the", "note")
    svg.text(440, 355, "dropped patches", "note")
    svg.path("M464,284 H488")
    svg.image(488, 248, 72, "decoded", ex["decoded"])
    svg.tile_label(488, 248, 72, "0.12")
    svg.text(524, 344, '<tspan class="m">x&#770;</tspan>')
    svg.path("M560,284 H584")
    svg.add('  <rect class="box" x="584" y="256" width="88" height="56" rx="6"/>\n')
    svg.snowflake(660, 268)
    svg.text(628, 282, "SDXL", "bold")
    svg.text(628, 298, "refiner", "note")
    svg.text(628, 340, "caption-guided", "note")
    svg.text(628, 355, "refinement", "note")
    svg.path("M672,284 H696")
    if ex["refined"] is not None:
        svg.image(696, 248, 72, "refined", ex["refined"])
        svg.tile_label(696, 248, 72, ex["refined_bpp"])
    else:
        svg.placeholder(696, 248, 72, "not run")
    svg.text(732, 344, '<tspan class="m">x&#771;</tspan>')

    caption = _short(ex["caption"], 60)
    svg.path("M60,248 V216 H228", "side")
    svg.add('  <rect x="228" y="204" width="392" height="24" rx="4" fill="#FFFFFF" stroke="#333333"/>\n')
    svg.text(424, 221, f'"{caption}"', "note")
    svg.text(232, 198, f"+ one caption by BLIP ({ex['caption_bytes']} bytes)", "note", "start")
    svg.path("M620,216 H628 V256", "side")
    svg.write(out_path)


def make_pipeline_svg(ex, out_path):
    """Full method figure: encoder on top, decoder at the bottom (U-shape)."""
    svg = Svg(1192, 600, "TextMAE pipeline",
              "Patch scores select 144 of 196 patches; a ViT encoder and g_a map them to a latent that is "
              "quantized and arithmetic-coded with a hyperprior and channel-wise context model. Patch "
              "positions are Huffman-coded. The decoder mirrors the encoder and fills dropped patches with "
              "mask tokens. A BLIP caption of the input guides an SDXL refiner that produces the final image.",
              top=48)
    svg.panel(16, 64, 384, 208, "(1) Patch selection")
    svg.panel(16, 296, 384, 248, "(3) Text guidance")
    svg.panel(416, 64, 760, 480, "(2) MAE codec")

    # (1) x -> s -> x_K   (centre y = 168)
    svg.image(36, 132, 72, "input", ex["input"])
    svg.text(72, 226, '<tspan class="m">x</tspan>')
    svg.path("M108,168 H156")
    svg.text(132, 160, "score", "note")
    svg.image(156, 132, 72, "score", ex["score"])
    svg.text(192, 226, '<tspan class="m">s</tspan>')
    svg.path("M228,168 H276")
    svg.text(252, 160, "select", "note")
    svg.image(276, 132, 72, "kept", ex["kept"])
    svg.text(312, 226, '<tspan class="m">x<tspan class="ms" dy="4">K</tspan></tspan>')
    svg.text(208, 258, "texture x structure score, keep 144 of 196", "note")

    # (2) encoder row (centre y = 168)
    svg.path("M348,168 H440")
    svg.trapezoid(440, 116, 80, 104, 20, True, "enc")
    svg.text(480, 164, "ViT", "bold")
    svg.text(480, 180, "Encoder", "bold")
    svg.path("M520,168 H552")
    svg.trapezoid(552, 136, 48, 64, 12, True, "enc")
    svg.text(576, 174, 'g<tspan class="ms" dy="4">a</tspan>', "m")
    svg.path("M600,168 H640")
    svg.text(634, 160, "y", "m")
    svg.add('  <rect class="box" x="640" y="152" width="32" height="32"/>\n')
    svg.text(656, 173, "Q", "bold")
    svg.path("M672,168 H704")
    svg.add('  <rect class="box" x="704" y="152" width="48" height="32"/>\n')
    svg.text(728, 173, "AE", "bold")

    # bitstream of y
    svg.path("M728,184 V248")
    svg.strip(721, 248, 12, vertical=True)
    svg.text(746, 312, "bits of <tspan class=\"ms\">y</tspan>", "note", "start")
    svg.path("M728,368 V424")
    svg.add('  <rect class="box" x="704" y="424" width="48" height="32"/>\n')
    svg.text(728, 445, "AD", "bold")

    # entropy model: y in, probabilities out, z coded and decoded again at the receiver
    svg.path("M620,168 V116 H888 V248")
    svg.add('  <circle cx="620" cy="168" r="3" fill="#333333"/>\n')
    svg.add('  <rect class="box" x="800" y="248" width="176" height="104" rx="8"/>\n')
    svg.text(888, 274, "Entropy model", "bold")
    svg.text(888, 296, "hyperprior <tspan class=\"ms\">z</tspan>", "note")
    svg.text(888, 314, "channel-wise context", "note")
    svg.text(888, 332, "(12 slices) + LRP", "note")
    svg.path("M800,280 H776 V168 H752", "prob")
    svg.path("M800,336 H776 V440 H752", "prob")
    svg.text(782, 230, '<tspan class="ms">&#956;, &#963;</tspan>', "note", "start")
    svg.path("M976,308 H1000")
    svg.strip(1000, 301, 6)
    svg.text(1030, 294, "bits of <tspan class=\"ms\">z</tspan>", "note")
    svg.path("M1030,315 V384 H888 V352")
    svg.text(959, 378, "decoded <tspan class=\"ms\">z&#770;</tspan>", "note")

    # patch positions (Huffman-coded side information)
    svg.path("M312,132 V100 H1112 V136", "side")
    svg.add('  <rect class="box" x="1064" y="136" width="96" height="48" rx="4"/>\n')
    svg.text(1112, 157, "Huffman", "bold")
    svg.text(1112, 174, "patch positions", "note")
    svg.path("M1112,184 V208")
    svg.strip(1082, 208, 6)
    svg.path("M1112,222 V520 H480 V508", "side")
    svg.text(796, 514, "patch positions: where the mask tokens go", "note")

    # (2) decoder row (centre y = 440)
    svg.path("M704,440 H600")
    svg.text(656, 432, "y&#770;", "m")
    svg.trapezoid(552, 408, 48, 64, 12, False, "dec")
    svg.text(576, 446, 'g<tspan class="ms" dy="4">s</tspan>', "m")
    svg.path("M552,440 H520")
    svg.trapezoid(440, 388, 80, 104, 20, False, "dec")
    svg.text(480, 436, "ViT", "bold")
    svg.text(480, 452, "Decoder", "bold")

    # (3) text guidance
    svg.path("M440,440 H384")
    svg.image(304, 400, 80, "decoded", ex["decoded"])
    svg.tile_label(304, 400, 80, "0.12")
    svg.text(344, 502, '<tspan class="m">x&#770;</tspan>')
    svg.path("M304,440 H272")
    svg.add('  <rect class="box" x="168" y="408" width="104" height="64" rx="6"/>\n')
    svg.snowflake(258, 420)
    svg.text(220, 436, "SDXL", "bold")
    svg.text(220, 454, "refiner", "note")
    svg.path("M168,440 H120")
    if ex["refined"] is not None:
        svg.image(40, 400, 80, "refined", ex["refined"])
        svg.tile_label(40, 400, 80, ex["refined_bpp"])
    else:
        svg.placeholder(40, 400, 80, "not run")
    svg.text(80, 502, '<tspan class="m">x&#771;</tspan>')
    svg.text(80, 522, "final image", "note")

    svg.path("M36,168 H26 V344 H168")
    svg.add('  <rect class="box" x="168" y="320" width="104" height="48" rx="6"/>\n')
    svg.snowflake(258, 332)
    svg.text(220, 342, "BLIP", "bold")
    svg.text(220, 358, "captioner", "note")
    svg.path("M220,368 V408", "side")
    svg.text(232, 392, '<tspan class="ms">c</tspan> (caption text)', "note", "start")

    # Legend
    svg.add('  <g transform="translate(24,568)">\n')
    svg.add('  <polygon class="enc" points="0,0 18,4 18,12 0,16"/>\n')
    svg.text(26, 13, "encoder (trained)", "note", "start")
    svg.add('  <polygon class="dec" points="140,4 158,0 158,16 140,12"/>\n')
    svg.text(166, 13, "decoder (trained)", "note", "start")
    svg.snowflake(290, 8)
    svg.text(302, 13, "frozen, pretrained", "note", "start")
    svg.strip(420, 1, 4, cell=8, thick=12)
    svg.text(458, 13, "bitstream", "note", "start")
    svg.path("M536,8 H572")
    svg.text(580, 13, "data", "note", "start")
    svg.path("M624,8 H660", "side")
    svg.text(668, 13, "side information (sent)", "note", "start")
    svg.path("M836,8 H872", "prob")
    svg.text(880, 13, "probabilities", "note", "start")
    svg.add('  </g>\n')
    svg.write(out_path)


def make_refinement_figure(refined, meta, out_path):
    """Original vs decoded (0.12 bpp) vs caption-guided refinement, with a zoom on one head."""
    side = 512
    original = np.array(Image.open(ROOT / "datasets" / "kodak" / "kodim23.png").convert("RGB")
                        .resize((side, side), Image.BICUBIC))
    decoded = np.array(Image.fromarray(decoded_rgb()).resize((side, side), Image.BICUBIC))
    refined = np.array(Image.fromarray(refined).resize((side, side), Image.LANCZOS))
    # Drop the top band (blurred background) that holds the burned-in label of the source figure
    top = int(np.ceil(TILE_LABEL[3] * side))
    original, decoded, refined = original[top:], decoded[top:], refined[top:]
    x0, y0, x1, y1 = (int(f * side) for f in (0.06, 0.32, 0.40, 0.66))  # left parrot's head
    y0, y1 = y0 - top, y1 - top

    columns = [
        (original, "Original"),
        (decoded, "Decoded $\\hat{x}$ (0.12 bpp)"),
        (refined, f"Refined $\\tilde{{x}}$ (+ caption, {meta['caption_bpp_at_224']:.3f} bpp)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), dpi=130,
                             gridspec_kw={"height_ratios": [(side - top) / side, 1]})
    for col, (img, title) in enumerate(columns):
        upper, bottom = axes[0, col], axes[1, col]
        upper.imshow(img)
        upper.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#2B6CB0", lw=1.8))
        upper.set_title(title, fontsize=12, color=INK)
        bottom.imshow(img[y0:y1, x0:x1], interpolation="lanczos")
        for ax in (upper, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#c8c8c8")
        for spine in bottom.spines.values():
            spine.set_edgecolor("#2B6CB0")
            spine.set_linewidth(1.8)
    fig.text(0.5, 0.015,
             f'Caption sent with the bitstream (BLIP): "{meta["caption"]}"  |  SDXL refiner, strength '
             f'{meta["strength"]}, {meta["timings_s"]["refine_s"]:.0f} s on {meta["device"].upper()}',
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    Image.open(buf).convert("RGB").save(out_path, quality=90, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Regenerate the README figures")
    parser.add_argument("--image", default=str(ROOT / "datasets" / "kodak" / "kodim23.png"))
    args = parser.parse_args()

    out_dir = ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb, s_map, t_map, scores = load_inputs(args.image)
    make_patch_figure(rgb, s_map, t_map, scores, out_dir / "patch_selection.png")

    refined, meta = refined_example()
    example = {
        "input": rgb,
        "score": score_heatmap(scores),
        "kept": masked_view(rgb, kept_mask(scores, "stratified")),
        "decoded": decoded_rgb(),
        "refined": refined,
        "caption": meta["caption"] if meta else "caption",
        "caption_bytes": meta["caption_bytes"] if meta else "a few",
        # total rate of the refined image: decoded image (0.12 bpp) + caption
        "refined_bpp": f"{0.12 + meta['caption_bpp_at_224']:.2f}" if meta else "",
    }
    make_concept_svg(example, out_dir / "concept.svg")
    make_pipeline_svg(example, out_dir / "pipeline.svg")
    make_rd_figure(out_dir / "rd_kodak.png")
    if refined is not None:
        make_refinement_figure(refined, meta, out_dir / "refinement.jpg")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
