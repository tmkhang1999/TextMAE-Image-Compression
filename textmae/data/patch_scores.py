"""Turn texture / structure maps into one importance score per patch."""
import cv2
import numpy as np

from textmae.data.score_maps import structure_map, texture_map


def patch_means(score_map, patch_size=16):
    """Integer mean of every non-overlapping patch, in row-major order (shape: (L,))."""
    h, w = score_map.shape
    return np.array([
        int(score_map[y: y + patch_size, x: x + patch_size].mean())
        for y in range(0, h - patch_size + 1, patch_size)
        for x in range(0, w - patch_size + 1, patch_size)
    ])


def image_patch_scores(gray, input_size=224, patch_size=16):
    """
    Importance of every patch of a grayscale image, min-max normalised to [0, 1].

    score = texture score * structure score, computed on an `input_size` square grid.

    NOTE: the texture map is taken from the image AFTER `structure_map` binarised it in place.
    That is how the shipped score files were generated, so it is kept for reproducibility.
    """
    size = (input_size, input_size)
    gray = gray.copy()

    s_map = structure_map(gray, size)
    t_map = texture_map(gray, size)

    total = patch_means(t_map, patch_size) * patch_means(s_map, patch_size)
    if total.size > 0:
        span = total.max() - total.min()
        total = (total - total.min()) / span if span > 0 else np.zeros_like(total, dtype=np.float64)
    return total.astype(np.float32)


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


def show_score_map(score, shape=(14, 14), title=""):
    """Debug helper: plot a patch score vector as a grid."""
    import matplotlib.pyplot as plt

    plt.imshow(np.resize(score, shape))
    plt.title(title, fontsize=16)
    plt.axis("off")
