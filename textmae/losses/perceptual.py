"""VGG16 feature loss (relu2_2 + relu3_3), computed on the [-1, 1] image range."""
from collections import namedtuple

import torch.nn as nn
from torchvision import models

from textmae.utils.image import de_normalize, normalize_batch

VggOutputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

# Slice boundaries of vgg16.features that end at relu1_2 / 2_2 / 3_3 / 4_3.
_VGG_SLICES = ((0, 4), (4, 9), (9, 16), (16, 23))

_vgg_cache = {}


class Vgg16(nn.Module):
    """Frozen ImageNet VGG16 that returns intermediate feature maps."""

    def __init__(self):
        super().__init__()
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        for start, end in _VGG_SLICES:
            block = nn.Sequential()
            for idx in range(start, end):
                block.add_module(str(idx), features[idx])
            self.slices.append(block)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        outputs = []
        for block in self.slices:
            x = block(x)
            outputs.append(x)
        return VggOutputs(*outputs)


def get_vgg(device):
    """Build the VGG16 once per device instead of on every loss call."""
    key = str(device)
    if key not in _vgg_cache:
        _vgg_cache[key] = Vgg16().to(device)
    return _vgg_cache[key]


def feature_loss(preds, imgs):
    """
    MSE between VGG16 relu2_2 and relu3_3 features of the prediction and target.

    Args:
        preds (torch.Tensor): Reconstruction, shape (N, 3, H, W), ImageNet-normalised.
        imgs (torch.Tensor): Target image, same shape and range.
    """
    vgg = get_vgg(preds.device)
    pred_feat = vgg(normalize_batch(de_normalize(preds)))
    gt_feat = vgg(normalize_batch(de_normalize(imgs)))
    mse = nn.MSELoss()
    return mse(pred_feat.relu2_2, gt_feat.relu2_2) + mse(pred_feat.relu3_3, gt_feat.relu3_3)
