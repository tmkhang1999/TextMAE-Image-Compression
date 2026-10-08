"""Shared inputs of the figure scripts: the example image, its patch scores and the decoded / refined results."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.patch_selection import select_patches  # noqa: E402
from src.scores import image_patch_scores, structure_map, texture_map  # noqa: E402

INPUT_SIZE, PATCH, KEEP = 224, 16, 144
GRID = INPUT_SIZE // PATCH
DROP_GRAY = (138, 136, 126)  # same gray as the masked inputs in assets/1.png
BLUE, TEAL, GRAY, INK = "#2B6CB0", "#2C7A7B", "#8A8A8A", "#172033"


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



# Bottom-right tile of assets/2.png: the 0.12 bpp / 27.5 dB reconstruction of kodim23. Its top-right corner carries a
# burned-in "bpp / dB" label, which the figures cover with their own label box.
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
