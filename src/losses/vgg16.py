"""VGG16 feature loss (relu2_2 + relu3_3) for images in the [-1, 1] range."""
import torch
import torch.nn as nn
from torchvision import models

_SLICES = ((0, 4), (4, 9), (9, 16), (16, 23))  # vgg16.features up to relu1_2 / 2_2 / 3_3 / 4_3
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_cache = {}


class Vgg16(nn.Module):
    """Frozen ImageNet VGG16 returning the four intermediate feature maps."""

    def __init__(self):
        super().__init__()
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList(nn.Sequential(*features[a:b]) for a, b in _SLICES)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        out = []
        for block in self.slices:
            x = block(x)
            out.append(x)
        return out


def vgg_input(x):
    """[-1, 1] -> ImageNet-normalised."""
    return ((x + 1.0) / 2.0 - _MEAN.to(x.device)) / _STD.to(x.device)


def feature_loss(preds, imgs):
    """MSE between the VGG16 relu2_2 and relu3_3 features of the prediction and the target."""
    key = str(preds.device)
    if key not in _cache:  # build the network once per device
        _cache[key] = Vgg16().to(preds.device)
    p, t = _cache[key](vgg_input(preds)), _cache[key](vgg_input(imgs))
    return sum(nn.functional.mse_loss(p[i], t[i]) for i in (1, 2))
