"""SVG diagrams in the HyRES style: the overview of the method and the detail of the MAE codec / LIC entropy model."""
from tools.figures.common import TILE_LABEL, jpeg_data_uri

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
    "lic": ("#d3e4ff", "#0b4fbf"),
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
            "compresses to 0.12 bits per pixel: an MAE (ViT) backbone fills in the dropped patches and a learned "
            "image compression (LIC) part produces the bitstream. A BLIP caption of the input costs 0.009 bits per pixel "
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
    f.line(f"M446 {cy} H492")

    # Learned codec: an MAE (ViT) backbone and the LIC part that produces the bits
    f.rect(494, cy - 68, 140, 136, "learned", rx=12, dashed=True)
    f.title(564, cy - 49, "Learned codec", size=12.5)
    f.box(504, cy - 38, 120, 42, "learned", "MAE (ViT)", ["fills dropped patches"])
    f.line(f"M564 {cy + 6} V{cy + 18}", arrow=False)
    f.box(504, cy + 18, 120, 42, "lic", "LIC", ["makes the bitstream"])
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
    f.swatch(140, 410, "learned", "MAE, trained here")
    f.swatch(300, 410, "lic", "LIC, trained here")
    f.swatch(450, 410, "pretrained", "frozen, used as-is")
    f.write(out_path)


def make_codec_svg(ex, out_path):
    """Detail view: (a) MAE codec main path, (b) entropy model."""
    f = Fig(1000, 506, "TextMAE codec",
            "(a) The learned codec: an MAE (ViT) encoder and g_a map the kept patches to a latent y that is quantized "
            "and arithmetic-coded; g_s and a ViT decoder reconstruct the image, inserting mask tokens at "
            "the dropped positions given by the Huffman-coded patch positions. (b) The entropy model: a "
            "hyperprior and a channel-wise context model predict a Gaussian for every latent.")
    f.title(16, 22, "(a)  Learned codec: MAE (ViT) + LIC", size=13, anchor="start")
    f.title(742, 22, "(b)  Entropy model (LIC)", size=13, anchor="start")

    bw, bh, gap, x0 = 44, 100, 10, 84
    xs = [x0 + i * (bw + gap) for i in range(4)]

    def group(x_from, x_to, y, kind, label):
        """Colored bar with a label above (or below) a run of blocks: which part is MAE, which is LIC."""
        stroke = KINDS[kind][1]
        f.add(f'<rect x="{x_from}" y="{y}" width="{x_to - x_from}" height="3" rx="1.5" fill="{stroke}"/>')
        f.text((x_from + x_to) / 2, y - 6, label, size=11.5, weight="bold", fill=stroke)

    def block(x, y, label, kind="learned"):
        f.rect(x, y, bw, bh, kind, rx=6)
        f.text(x + bw / 2, y + bh / 2, label, size=11, fill=INK, rotate=True)

    # ---- encoder row (centre y = 112)
    ey = 62
    f.title(340, 50, "Encoder", size=12.5, anchor="start")
    f.text(398, 50, "runs at the sender", size=11, anchor="start")
    group(xs[0], xs[1] + bw, 48, "learned", "MAE (ViT)")
    group(xs[2], xs[3] + bw, 48, "lic", f'LIC, analysis {math("g", "a")}')
    f.text(40, 105, math("x", "K"), size=16, fill=INK)
    f.text(40, 124, "kept patches", size=10)
    f.line(f"M66 {ey + 50} H{xs[0] - 2}")
    for x, label, kind in zip(xs, ["Patch embed", "ViT blocks x12", "Reshape to grid", "Conv 1x1 x4"],
                              ["learned", "learned", "lic", "lic"]):
        block(x, ey, label, kind)
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
    f.title(340, 236, "Decoder", size=12.5, anchor="start")
    f.text(398, 236, "runs at the receiver", size=11, anchor="start")
    group(xs[0], xs[2] + bw, 236, "learned", "MAE (ViT)")
    group(xs[3], xs[3] + bw, 236, "lic", f'LIC, {math("g", "s")}')
    f.rect(qx - 30, dy + 34, 60, 32, "entropy", rx=16)
    f.title(qx, dy + 55, "AD", size=12.5)
    f.line(f"M{qx - 30} {dy + 50} H{xs[3] + bw + 4}")
    f.text(420, dy + 40, math("y", hat="^"), size=16, fill=INK)
    for x, label, kind in zip(reversed(xs), ["ConvT 1x1 x4", "Mask tokens", "ViT blocks x8", "Predict patches"],
                              ["lic", "learned", "learned", "learned"]):
        block(x, dy, label, kind)
    f.text(xs[3] + bw / 2, dy + bh + 16, "12 x 12 x 768", size=9.5)
    f.text(xs[0] + bw / 2, dy + bh + 16, "224 x 224 x 3", size=9.5)
    f.line(f"M{xs[0]} {dy + 50} H66")
    f.text(40, dy + 44, math("x", hat="^"), size=16, fill=INK)

    # ---- entropy model box shared by encoder and decoder
    ex0, ew = 650, 70
    f.rect(ex0, ey + 34, ew, dy + 66 - ey - 34, "lic")
    f.text(ex0 + ew / 2 - 6, 210, "Entropy model", size=13, weight="bold", fill=INK, rotate=True)
    f.text(ex0 + ew / 2 + 12, 210, "used by encoder and decoder", size=10.5, rotate=True)
    f.line(f"M{ex0} {ey + 104} H{qx + 36}")
    f.text(620, ey + 96, f'{math("&#956;", None)}, {math("&#963;", None)}', size=12, fill=INK)
    f.line(f"M{ex0} {dy + 50} H{qx + 36}")
    f.text(620, dy + 42, f'{math("&#956;", None)}, {math("&#963;", None)}', size=12, fill=INK)
    f.line(f"M500 {ey + 50} V{ey - 8} H{ex0 + ew / 2} V{ey + 32}")
    f.dot(qx - 60, dy + 50)
    f.line(f"M{qx - 60} {dy + 50} V{dy + 124} H{ex0 + ew / 2} V{dy + 68}", dashed=True)
    f.text(640, dy + 140, "decoded slices", size=11)

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
    f.box(bx, 72, w, 40, "lic", f'Hyper analysis {math("h", "a")}')
    f.line(f"M{cx} 112 V138")
    f.text(cx + 14, 130, math("z"), size=14, fill=INK, anchor="start")
    f.box(bx, 140, w, 52, "entropy", "Q, AE / AD", ["factorized prior"])
    f.line(f"M{cx} 192 V218")
    f.box(bx, 220, w, 40, "lic", f'Hyper synthesis {math("h", "s")}')
    f.line(f"M{cx} 260 V286")
    f.box(bx, 288, w, 66, "lic", "Channel-wise context", ["12 slices of y, Conv 3x3 stacks", "+ latent residual prediction"])
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
    f.swatch(16, 484, "learned", "MAE (ViT)", w=26)
    f.swatch(130, 484, "lic", "LIC (learned image compression)", w=26)
    f.swatch(360, 484, "entropy", "entropy coding", w=26)
    f.swatch(500, 484, "pretrained", "quantizer", w=26)
    f.text(620, 496, "AE / AD: arithmetic encoder / decoder", size=10.5, anchor="start")
    f.write(out_path)
