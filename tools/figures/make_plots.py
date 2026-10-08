"""Matplotlib figures: patch scoring and selection, rate-distortion vs JPEG / WebP, and the refinement comparison."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tools.figures.common import (GRAY, GRID, INK, INPUT_SIZE, KEEP, ROOT, TEAL, BLUE, TEXTMAE_POINTS, TILE_LABEL,  # noqa: E402
                                  decoded_rgb, kept_mask, masked_view)


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
