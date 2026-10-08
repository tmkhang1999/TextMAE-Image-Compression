"""Per-patch importance scores: a structure score (quad-tree split-and-merge segmentation) times a texture score
(absolute Laplacian), averaged per patch and min-max normalised. Smooth patches score low.

    python -m src.scores --training_path datasets/train_dataset --testing_path datasets/test_dataset

writes <dataset>_scores/{train,val,test}.pt: one row of (input_size / patch_size) ** 2 scores per image,
in `collect_images` order.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.utils.dataset_utils import collect_images, scores_path, split_root


def _homogeneous(img, h0, w0, h, w):
    """True if at least 95% of the block is within 2 std of its mean (no need to split)."""
    area = img[h0:h0 + h, w0:w0 + w]
    return np.count_nonzero((area - np.mean(area)) < 2 * np.std(area, ddof=1)) / area.size >= 0.95


def _split_and_merge(img, h0, w0, h, w):
    """Split non-homogeneous blocks into quadrants; binarise the leaves in place (mid gray -> 0, else 255)."""
    if not _homogeneous(img, h0, w0, h, w) and min(h, w) > 5:
        hh, hw = int(h / 2), int(w / 2)
        for dy, dx in ((0, 0), (0, hw), (hh, 0), (hh, hw)):
            _split_and_merge(img, h0 + dy, w0 + dx, hh, hw)
    else:
        area = img[h0:h0 + h, w0:w0 + w]
        mask = (60 < area) & (area < 150)
        area[mask], area[~mask] = 0, 255


def structure_map(gray, size):
    """Split-and-merge segmentation resized to `size`. NOTE: `gray` is binarised in place."""
    _split_and_merge(gray, 0, 0, gray.shape[0], gray.shape[1])
    return cv2.resize(gray[1:-1, 1:-1], size)  # drop the 1px border the recursion covers unevenly


def texture_map(gray, size):
    return cv2.resize(cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3)), size)


def patch_means(score_map, patch_size=16):
    """Integer mean of every non-overlapping patch, row-major."""
    h, w = score_map.shape
    return np.array([int(score_map[y:y + patch_size, x:x + patch_size].mean())
                     for y in range(0, h - patch_size + 1, patch_size) for x in range(0, w - patch_size + 1, patch_size)])


def image_patch_scores(gray, input_size=224, patch_size=16):
    """Scores in [0, 1] of every patch of a grayscale image on an `input_size` square grid."""
    size = (input_size, input_size)
    structure = structure_map(gray.copy(), size)
    total = patch_means(texture_map(gray, size), patch_size) * patch_means(structure, patch_size)
    span = total.max() - total.min()
    return ((total - total.min()) / span if span > 0 else np.zeros(total.shape)).astype(np.float32)


def generate(mode, dataset_path, input_size, patch_size):
    paths = collect_images(split_root(dataset_path, mode))
    if not paths:
        raise RuntimeError(f"No images found in {split_root(dataset_path, mode)}")
    rows = []
    for path in tqdm(paths, desc=f"{Path(dataset_path).name}/{mode}"):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Could not read image: {path}")
        rows.append(torch.from_numpy(image_patch_scores(gray, input_size, patch_size)))
    out = scores_path(dataset_path, mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(rows), out)
    print(f"Saved {out} {tuple(torch.stack(rows).shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_path", required=True, help="Folder with train/ and val/ images")
    parser.add_argument("--testing_path", required=True, help="Folder with test images")
    parser.add_argument("--input_size", type=int, default=224, help="Must match training and evaluation")
    parser.add_argument("--patch_size", type=int, default=16)
    args = parser.parse_args()
    for mode in ("train", "val"):
        generate(mode, args.training_path, args.input_size, args.patch_size)
    generate("test", args.testing_path, args.input_size, args.patch_size)
