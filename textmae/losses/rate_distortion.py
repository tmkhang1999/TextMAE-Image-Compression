import math

import torch.nn as nn


class RateDistortionLoss(nn.Module):
    """Rate-distortion objective: lmbda * distortion + bits-per-pixel."""

    # Weights of the three distortion terms returned by TextMAE.forward_loss.
    SSIM_WEIGHT = 0.25
    L1_WEIGHT = 10.0
    VGG_WEIGHT = 0.1

    def __init__(self, lmbda=1e-2):
        """
        Args:
            lmbda (float): Trade-off between distortion and rate.
        """
        super().__init__()
        self.lmbda = lmbda

    def forward(self, output, target):
        """
        Args:
            output (dict): Model output with "likelihoods" (dict of tensors) and
                "loss" (ssim_loss, l1_loss, vgg_loss).
            target (torch.Tensor): Input batch, shape (N, 3, H, W).

        Returns:
            dict: bpp_loss, ssim_loss, L1_loss, vgg_loss and the combined loss.
        """
        N, _, H, W = target.size()
        num_pixels = N * H * W

        out = {}
        out["bpp_loss"] = sum(
            (likelihoods.log().sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["ssim_loss"], out["L1_loss"], out["vgg_loss"] = output["loss"]

        distortion = (
            self.SSIM_WEIGHT * out["ssim_loss"]
            + self.L1_WEIGHT * out["L1_loss"]
            + self.VGG_WEIGHT * out["vgg_loss"]
        )
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        return out
