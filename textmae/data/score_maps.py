"""
Texture and structure maps used to rate how important each image patch is.

- structure: quad-tree split-and-merge segmentation of the grayscale image
- texture:   Laplacian (edge) response
"""
import cv2
import numpy as np


def _is_homogeneous(img, h0, w0, h, w):
    """True if at least 95% of the block is within 2 std of its mean (no need to split)."""
    area = img[h0: h0 + h, w0: w0 + w]
    mean = np.mean(area)
    std = np.std(area, ddof=1)
    return np.count_nonzero((area - mean) < 2 * std) / area.size >= 0.95


def _merge(img, h0, w0, h, w):
    """Binarise a block in place: mid-gray pixels -> 0, everything else -> 255."""
    area = img[h0:h0 + h, w0:w0 + w]
    mask = (60 < area) & (area < 150)
    img[h0:h0 + h, w0:w0 + w][mask] = 0
    img[h0:h0 + h, w0:w0 + w][~mask] = 255


def _split_and_merge(img, h0, w0, h, w):
    """Recursively split non-homogeneous blocks into quadrants, then binarise the leaves."""
    if not _is_homogeneous(img, h0, w0, h, w) and min(h, w) > 5:
        half_h, half_w = int(h / 2), int(w / 2)
        _split_and_merge(img, h0, w0, half_h, half_w)
        _split_and_merge(img, h0, w0 + half_w, half_h, half_w)
        _split_and_merge(img, h0 + half_h, w0, half_h, half_w)
        _split_and_merge(img, h0 + half_h, w0 + half_w, half_h, half_w)
    else:
        _merge(img, h0, w0, h, w)


def structure_map(gray, size):
    """
    Split-and-merge segmentation resized to `size` (width, height).

    NOTE: `gray` is binarised IN PLACE; pass a copy if the original is still needed.
    """
    _split_and_merge(gray, 0, 0, gray.shape[0], gray.shape[1])
    # Drop the 1px border, which the recursion leaves unevenly covered
    return cv2.resize(gray[1:-1, 1:-1], size)


def texture_map(gray, size):
    """Absolute Laplacian response resized to `size` (width, height)."""
    laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    return cv2.resize(cv2.convertScaleAbs(laplacian), size)
