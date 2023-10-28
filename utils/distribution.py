import numpy as np
import matplotlib.pyplot as plt


def cal_patch_score(img, crop_sz=16, step=16):
    h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)

    patch_scores = [
        int(img[x : x + crop_sz, y : y + crop_sz].mean())
        for x in h_space
        for y in w_space
    ]

    return np.array(patch_scores)


def show_distribution(score, shape=(14, 14), title=""):
    score = np.resize(score, shape)
    plt.imshow(score)
    plt.title(title, fontsize=16)
    plt.axis("off")
    return
