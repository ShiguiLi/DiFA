"""BCHW reference geometry. No network or solver dependency."""
import math
import torch
import torch.nn.functional as F

def image_to_patch_channels(image, patch_size):
    b, channels, height, width = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError('Image dimensions must be divisible by patch_size')
    return image.reshape(b, channels, height // patch_size, patch_size, width // patch_size, patch_size).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, channels * patch_size ** 2)
def patch_to_image(patches, height, width, p):
    b, _, dim = patches.shape
    c = dim // (p * p)
    return patches.reshape(b, height // p, width // p, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, height, width)
def difa_linear(current, residuals, ell, scale=1.75):
    """D_e,t on [B,K,C,H,W]. The spatial weights, if any, are already inside.

    This is the inspected DiFA guidance transform (not the output prediction).
    The projection is global per image; the 5x5 filter follows the projection.
    """
    e = current[:, None]
    dims = (2, 3, 4)
    radial = (residuals * e).sum(dims, keepdim=True) / (e.square().sum(dims, keepdim=True) + 1e-08)
    perpendicular = residuals - radial * e
    b, k, c, h, w = perpendicular.shape
    low = F.avg_pool2d(perpendicular.reshape(b * k, c, h, w), 5, 1, 2).reshape(b, k, c, h, w)
    high_weight = 1.0 / (1.0 + math.exp(-ell))
    return (scale - 1.0) * (low + high_weight * (perpendicular - low))

def to_tokens(images, p):
    b, k, c, h, w = images.shape
    tokens = image_to_patch_channels(images.reshape(b * k, c, h, w), p)
    return tokens.reshape(b, k, tokens.shape[1], tokens.shape[2])
