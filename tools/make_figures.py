"""
Regenerate the README figures from the real scoring / selection code.

    python tools/make_figures.py

Writes to assets/figures/:
    overview.svg          whole method in one row, bit rates on the arrows
    codec.svg             detail of the MAE codec and its entropy model
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
# Same visual language as the HyRES figures: Helvetica, rounded boxes, two accent colors
# (blue = learned, orange = handcrafted / entropy coding), real images with bit rates on
# the arrows. Pretrained models used as-is are drawn in white.

INK = "#172033"
MUTED = "#475467"
EDGE = "#d9e0e8"
BLUE_FILL, BLUE_EDGE = "#eaf3ff", "#1677ff"
ORANGE_FILL, ORANGE_EDGE = "#f6e3bd", "#b7791f"
KINDS = {  # box kind -> (fill, stroke)
    "learned": (BLUE_FILL, BLUE_EDGE),
    "handcrafted": (ORANGE_FILL, ORANGE_EDGE),
    "entropy": (ORANGE_FILL, ORANGE_EDGE),
    "pretrained": ("#ffffff", MUTED),
}
SANS = "Helvetica, Arial, sans-serif"
SERIF = "'Times New Roman', Times, serif"


def math(sym, sub=None, hat=None):
    """Italic serif symbol with an optional subscript, e.g. math('x', 'K') or math('x', hat='^')."""
    sym = {"^": sym + "&#770;", "~": sym + "&#771;"}.get(hat, sym)
    out = f'<tspan font-family="{SERIF}" font-style="italic">{sym}</tspan>'
    if sub:
        out += (f'<tspan font-family="{SERIF}" font-style="italic" font-size="75%" dy="3">{sub}</tspan>'
                f'<tspan dy="-3"></tspan>')
    return out


class Fig:
    """Collects SVG elements. Every image is embedded once in <defs> and referenced with <use>."""

    def __init__(self, width, height, title, desc, top=0):
        # `top` crops empty space above the content without moving every coordinate
        self.width, self.height, self.top = width, height, top
        self.doc_title, self.doc_desc = title, desc
        self.images, self.body = {}, []

    # --- primitives
    def add(self, element):
        self.body.append("  " + element + "\n")

    def text(self, x, y, s, size=11.5, weight="normal", fill=MUTED, anchor="middle", rotate=False):
        extra = f' transform="rotate(-90 {x} {y})" dominant-baseline="central"' if rotate else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{extra}>{s}</text>')

    def title(self, x, y, s, fill=INK, anchor="middle", size=14):
        self.text(x, y, s, size=size, weight="bold", fill=fill, anchor=anchor)

    def line(self, d, dashed=False, arrow=True, color=MUTED):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" '
                 f'stroke-linejoin="round"{dash}{marker}/>')

    def dot(self, x, y):
        self.add(f'<circle cx="{x}" cy="{y}" r="3.2" fill="{MUTED}"/>')

    def rect(self, x, y, w, h, kind, rx=12, dashed=False):
        fill, stroke = KINDS[kind]
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="1.7"{dash}/>')

    def box(self, x, y, w, h, kind, title, lines=(), dashed=False):
        """Rounded box with a bold title and muted description lines, centred."""
        self.rect(x, y, w, h, kind, dashed=dashed)
        cy = y + h / 2
        top = cy - (18 + 17.5 * len(lines)) / 2 + 13
        self.title(x + w / 2, top, title)
        for i, line in enumerate(lines):
            self.text(x + w / 2, top + 18 + 17.5 * i, line)

    def image(self, x, y, size, key, rgb, stroke=EDGE):
        if key not in self.images:
            self.images[key] = jpeg_data_uri(rgb)
        self.add(f'<use href="#img-{key}" xlink:href="#img-{key}" transform="translate({x},{y}) scale({size})"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="none" stroke="{stroke}" stroke-width="1.7"/>')

    def tag(self, x, y, size, text):
        """White label box over the corner of a result tile (covers the burned-in label of assets/2.png)."""
        x0, y0, x1, y1 = TILE_LABEL
        bx, by, bw, bh = x + x0 * size, y + y0 * size, (x1 - x0) * size, (y1 - y0) * size
        self.add(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#ffffff" '
                 f'stroke="{INK}" stroke-width="0.8"/>')
        self.text(round(bx + bw / 2, 1), round(by + bh / 2 + 3, 1), text, size=9.5, weight="bold", fill=INK)

    def strip(self, x, y, cells, cell=10, thick=14, vertical=False):
        """Bitstream icon: a fixed black / white pattern."""
        pattern = "1011001110100101110010110"
        for i in range(cells):
            fill = "#172033" if pattern[i % len(pattern)] == "1" else "#ffffff"
            cx, cy = (x, y + i * cell) if vertical else (x + i * cell, y)
            w, h = (thick, cell) if vertical else (cell, thick)
            self.add(f'<rect x="{cx}" y="{cy}" width="{w}" height="{h}" fill="{fill}" stroke="{INK}" stroke-width="0.8"/>')

    def swatch(self, x, y, kind, label, w=26):
        self.rect(x, y, w, 16, kind, rx=5)
        self.text(x + w + 8, y + 12, label, size=11.5, anchor="start")

    def write(self, out_path):
        images = "".join(
            f'    <image id="img-{k}" width="1" height="1" preserveAspectRatio="none" href="{uri}" xlink:href="{uri}"/>\n'
            for k, uri in self.images.items())
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
               f'viewBox="0 {self.top} {self.width} {self.height - self.top}" width="{self.width}" '
               f'height="{self.height - self.top}" '
               f'role="img" aria-labelledby="t d">\n'
               f'  <title id="t">{self.doc_title}</title>\n  <desc id="d">{self.doc_desc}</desc>\n'
               f'  <defs>\n    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
               f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>\n{images}  </defs>\n'
               f'  <rect y="{self.top}" width="{self.width}" height="{self.height}" fill="#ffffff"/>\n'
               + "".join(self.body) + "</svg>\n")
        out_path.write_text(svg, encoding="ascii")


def _short(text, limit):
    text = text.replace("&", "and").replace("<", "").replace(">", "").replace('"', "'")
    if len(text) <= limit:
        return text
    return text[:limit - 3].rsplit(" ", 1)[0] + "..."  # cut at a word boundary


def make_overview_svg(ex, out_path):
    """Whole method in one row: select, code, decode, refine; bit rates on the arrows."""
    f = Fig(1080, 440, "TextMAE overview",
            "The input image is reduced to its 144 most informative patches, which a learned MAE codec "
            "compresses to 0.12 bits per pixel. A BLIP caption of the input costs 0.009 bits per pixel "
            "and guides an SDXL refiner that turns the decoded image into the final image.", top=72)
    cy, size = 170, 104  # main row: centre line and image size
    top = cy - size // 2

    # Input
    f.title(68, top - 12, f'Input image {math("x")}')
    f.image(16, top, size, "input", ex["input"])
    f.text(68, top + size + 20, "Kodak kodim23")
    f.line(f"M126 {cy} H164")

    # Patch selection (handcrafted) -> kept patches
    f.box(168, cy - 40, 136, 80, "handcrafted", "Patch selection", ["texture x structure", "keep 144 of 196"])
    f.line(f"M306 {cy} H340")
    f.title(392, top - 12, f'Kept patches {math("x", "K")}')
    f.image(340, top, size, "kept", ex["kept"])
    f.text(392, top + size + 20, "gray = dropped")
    f.line(f"M446 {cy} H490")

    # Learned codec
    f.box(494, cy - 46, 140, 92, "learned", "MAE codec", ["ViT-B + entropy model", "(incl. patch positions)"])
    f.line(f"M636 {cy} H690")
    f.text(663, cy - 11, "0.12 bpp", size=11, weight="bold", fill=BLUE_EDGE)

    # Decoded image
    f.title(744, top - 12, f'Decoded {math("x", hat="^")}')
    f.image(692, top, size, "decoded", ex["decoded"])
    f.tag(692, top, size, "0.12")
    f.text(744, top + size + 20, "0.12 bpp, 27.5 dB")
    f.line(f"M798 {cy} H828")

    # Refiner (pretrained) -> final image
    f.box(830, cy - 40, 100, 80, "pretrained", "SDXL", ["refiner,", "caption-guided"])
    f.line(f"M932 {cy} H960")
    f.title(1012, top - 12, f'Final image {math("x", hat="~")}')
    f.image(960, top, size, "refined", ex["refined"])
    f.tag(960, top, size, ex["refined_bpp"])
    f.text(1012, top + size + 20, f'{ex["refined_bpp"]} bpp in total')

    # Text branch: input -> BLIP -> caption -> refiner
    by = 340
    f.line(f"M126 {cy + 22} C150 {cy + 22} 150 {by} 164 {by}")
    f.box(168, by - 40, 136, 80, "pretrained", "BLIP captioner", ["pretrained", "describes the input"])
    f.line(f"M306 {by} H350")
    f.rect(352, by - 24, 366, 48, "pretrained", rx=12, dashed=True)
    caption = _short(ex["caption"], 60)
    f.text(535, by - 2, f'"{caption}"', size=12, fill=INK)
    f.text(535, by + 14, f'caption {math("c")}, {ex["caption_bytes"]} bytes of text', size=11)
    f.line(f"M720 {by} H880 V{cy + 42}")
    f.text(800, by - 10, f'+{ex["caption_bpp"]:.3f} bpp', size=11, weight="bold", fill=INK)

    # Legend
    f.swatch(16, 410, "handcrafted", "handcrafted")
    f.swatch(150, 410, "learned", "learned, trained here")
    f.swatch(330, 410, "pretrained", "pretrained, used as-is")
    f.write(out_path)


def make_codec_svg(ex, out_path):
    """Detail view: (a) MAE codec main path, (b) entropy model."""
    f = Fig(1000, 506, "TextMAE codec",
            "(a) The MAE codec: a ViT encoder and g_a map the kept patches to a latent y that is quantized "
            "and arithmetic-coded; g_s and a ViT decoder reconstruct the image, inserting mask tokens at "
            "the dropped positions given by the Huffman-coded patch positions. (b) The entropy model: a "
            "hyperprior and a channel-wise context model predict a Gaussian for every latent.")
    f.title(16, 22, "(a)  MAE codec", size=13, anchor="start")
    f.title(742, 22, "(b)  Entropy model", size=13, anchor="start")

    bw, bh, gap, x0 = 44, 100, 10, 84
    xs = [x0 + i * (bw + gap) for i in range(4)]

    def block(x, y, label, kind="learned"):
        f.rect(x, y, bw, bh, kind, rx=6)
        f.text(x + bw / 2, y + bh / 2, label, size=11, fill=INK, rotate=True)

    # ---- encoder row (centre y = 112)
    ey = 62
    f.title(x0, 50, f'MAE encoder and analysis transform {math("g", "a")}', size=12.5, anchor="start")
    f.text(x0 + 330, 50, "runs in the encoder", size=11, anchor="start")
    f.text(40, 105, math("x", "K"), size=16, fill=INK)
    f.text(40, 124, "kept patches", size=10)
    f.line(f"M66 {ey + 50} H{xs[0] - 2}")
    for x, label in zip(xs, ["Patch embed", "ViT blocks x12", "Reshape to grid", "Conv 1x1 x4"]):
        block(x, ey, label)
    f.text(xs[0] + bw / 2, ey + bh + 16, "144 x 768", size=9.5)
    f.text(xs[3] + bw / 2, ey + bh + 16, "12 x 12 x 384", size=9.5)

    # y line -> Q -> AE -> bitstream -> AD
    qx = 560
    f.line(f"M{xs[3] + bw} {ey + 50} H{qx - 24}")
    f.text(400, ey + 40, math("y"), size=16, fill=INK)
    f.dot(500, ey + 50)
    f.rect(qx - 24, ey + 34, 48, 32, "pretrained", rx=8)
    f.title(qx, ey + 55, "Q", size=12.5)
    f.line(f"M{qx} {ey + 66} V{ey + 86}")
    f.rect(qx - 30, ey + 88, 60, 32, "entropy", rx=16)
    f.title(qx, ey + 109, "AE", size=12.5)
    f.line(f"M{qx} {ey + 120} V{ey + 138}")
    f.strip(qx - 7, ey + 140, 7, vertical=True)
    f.line(f"M{qx} {ey + 210} V{ey + 224}")

    # ---- decoder row (centre y = 302)
    dy = 252
    f.title(x0, 236, f'MAE decoder and synthesis transform {math("g", "s")}', size=12.5, anchor="start")
    f.text(x0 + 330, 236, "runs in the decoder", size=11, anchor="start")
    f.rect(qx - 30, dy + 34, 60, 32, "entropy", rx=16)
    f.title(qx, dy + 55, "AD", size=12.5)
    f.line(f"M{qx - 30} {dy + 50} H{xs[3] + bw + 4}")
    f.text(420, dy + 40, math("y", hat="^"), size=16, fill=INK)
    for x, label in zip(reversed(xs), ["ConvT 1x1 x4", "Mask tokens", "ViT blocks x8", "Predict patches"]):
        block(x, dy, label)
    f.text(xs[3] + bw / 2, dy + bh + 16, "12 x 12 x 768", size=9.5)
    f.text(xs[0] + bw / 2, dy + bh + 16, "224 x 224 x 3", size=9.5)
    f.line(f"M{xs[0]} {dy + 50} H66")
    f.text(40, dy + 44, math("x", hat="^"), size=16, fill=INK)

    # ---- entropy model box shared by encoder and decoder
    ex0, ew = 650, 70
    f.rect(ex0, ey + 34, ew, dy + 66 - ey - 34, "learned")
    f.text(ex0 + ew / 2 - 6, 210, "Entropy model", size=13, weight="bold", fill=INK, rotate=True)
    f.text(ex0 + ew / 2 + 12, 210, "used by encoder and decoder", size=10.5, rotate=True)
    f.line(f"M{ex0} {ey + 104} H{qx + 36}")
    f.text(620, ey + 96, f'{math("&#956;", None)}, {math("&#963;", None)}', size=12, fill=INK)
    f.line(f"M{ex0} {dy + 50} H{qx + 36}")
    f.text(620, dy + 42, f'{math("&#956;", None)}, {math("&#963;", None)}', size=12, fill=INK)
    f.line(f"M500 {ey + 50} V{ey - 8} H{ex0 + ew / 2} V{ey + 32}")
    f.dot(qx - 60, dy + 50)
    f.line(f"M{qx - 60} {dy + 50} V{dy + 124} H{ex0 + ew / 2} V{dy + 68}", dashed=True)
    f.text(600, dy + 140, "decoded slices", size=11)

    # ---- patch positions (side information), bottom row
    py = 430
    f.title(x0, py - 8, "Patch positions (side information)", size=12.5, anchor="start")
    f.box(x0, py, 126, 44, "handcrafted", "Selection", ["144 of 196 ids"])
    f.line(f"M{x0 + 126} {py + 22} H{x0 + 156}")
    f.box(x0 + 158, py, 104, 44, "entropy", "Huffman", ["encode"])
    f.line(f"M{x0 + 262} {py + 22} H{x0 + 290}")
    f.strip(x0 + 292, py + 15, 6)
    f.line(f"M{x0 + 354} {py + 22} H{x0 + 384}")
    f.box(x0 + 386, py, 104, 44, "entropy", "Huffman", ["decode"])
    f.line(f"M{x0 + 438} {py} V{py - 32} H{xs[2] + bw / 2} V{dy + bh + 4}", dashed=True)
    f.text(x0 + 470, py - 38, "where to put the mask tokens", size=11, anchor="end")

    # ---- (b) entropy model
    cx, w, bx = 860, 224, 748
    f.text(cx, 50, math("y"), size=16, fill=INK)
    f.line(f"M{cx} 56 V70")
    f.box(bx, 72, w, 40, "learned", f'Hyper analysis {math("h", "a")}')
    f.line(f"M{cx} 112 V138")
    f.text(cx + 14, 130, math("z"), size=14, fill=INK, anchor="start")
    f.box(bx, 140, w, 52, "entropy", "Q, AE / AD", ["factorized prior"])
    f.line(f"M{cx} 192 V218")
    f.box(bx, 220, w, 40, "learned", f'Hyper synthesis {math("h", "s")}')
    f.line(f"M{cx} 260 V286")
    f.box(bx, 288, w, 66, "learned", "Channel-wise context", ["12 slices of y, Conv 3x3 stacks", "+ latent residual prediction"])
    # slice icon
    cells, cell, sx = 12, 17, cx - 12 * 17 / 2
    for i in range(cells):
        fill, stroke = ("#ffffff", BLUE_EDGE)
        if i < 6:
            fill = "#b9d6ff"
        if i == 8:
            fill = BLUE_EDGE
        f.add(f'<rect x="{sx + i * cell:.1f}" y="372" width="{cell - 3}" height="{cell - 3}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    f.text(cx, 404, "slice i uses the hyperprior and", size=10.5)
    f.text(cx, 418, "the first min(i, 6) decoded slices (shaded)", size=10.5)
    f.line(f"M{cx} 428 V448")
    f.text(cx, 466, f'{math("&#956;", None)}, {math("&#963;", None)} for every latent', size=11.5, fill=INK)

    # legend
    f.swatch(16, 484, "learned", "learned module", w=26)
    f.swatch(160, 484, "entropy", "entropy coding", w=26)
    f.swatch(300, 484, "pretrained", "quantizer", w=26)
    f.text(440, 496, "Q: quantizer, AE / AD: arithmetic encoder / decoder", size=10.5, anchor="start")
    f.write(out_path)


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
        "caption_bpp": meta["caption_bpp_at_224"] if meta else 0.0,
        # total rate of the refined image: decoded image (0.12 bpp) + caption
        "refined_bpp": f"{0.12 + meta['caption_bpp_at_224']:.2f}" if meta else "",
    }
    make_overview_svg(example, out_dir / "overview.svg")
    make_codec_svg(example, out_dir / "codec.svg")
    make_rd_figure(out_dir / "rd_kodak.png")
    if refined is not None:
        make_refinement_figure(refined, meta, out_dir / "refinement.jpg")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
