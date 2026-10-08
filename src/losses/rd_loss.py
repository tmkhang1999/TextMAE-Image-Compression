import math

import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

from src.losses.vgg16 import feature_loss


class RateDistortionLoss(nn.Module):
    """lmbda * (0.25 * (1 - SSIM) + 10 * L1 + 0.1 * VGG feature loss) + bits per pixel."""
    SSIM_WEIGHT, L1_WEIGHT, VGG_WEIGHT = 0.25, 10.0, 0.1

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.lmbda = lmbda
        self.ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)

    def forward(self, output, target):
        N, _, H, W = target.size()
        x_hat = output["x_hat"]
        out = {
            "bpp_loss": sum(l.log().sum() / (-math.log(2) * N * H * W) for l in output["likelihoods"].values()),
            "ssim_loss": 1 - self.ssim(x_hat, target),
            "L1_loss": F.l1_loss(x_hat, target),
            "vgg_loss": feature_loss(x_hat, target),
        }
        distortion = (self.SSIM_WEIGHT * out["ssim_loss"] + self.L1_WEIGHT * out["L1_loss"]
                      + self.VGG_WEIGHT * out["vgg_loss"])
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        return out
