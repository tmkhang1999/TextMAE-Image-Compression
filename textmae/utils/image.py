"""Pixel-range conversions used by the perceptual (VGG) loss."""
import torch


def de_normalize(batch):
    """Map a [-1, 1] batch to the [0, 255] range."""
    return (batch + 1.0) / 2.0 * 255.0


def normalize_batch(batch):
    """Scale a [0, 255] batch to [0, 1] and apply the ImageNet mean/std."""
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= mean
    batch = batch / std
    return batch
