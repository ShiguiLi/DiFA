# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

@dataclass
class Refinement:
    """DiFA prediction-refinement primitive."""
    correction: torch.Tensor
    diagnostics: dict

def _to_groups(images, group_size):
    batch, count, channels, height, width = images.shape
    image = images.reshape(batch * count, channels, height, width)
    groups = image.reshape(batch * count, channels, height // group_size, group_size, width // group_size, group_size).permute(0, 2, 4, 1, 3, 5).reshape(batch * count, -1, channels * group_size ** 2)
    return groups.reshape(batch, count, groups.shape[1], groups.shape[2])

def _from_groups(groups, height, width, group_size):
    batch, _, dimension = groups.shape
    channels = dimension // (group_size * group_size)
    return groups.reshape(batch, height // group_size, width // group_size, channels, group_size, group_size).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)

class RefinementBase:
    """DiFA prediction-refinement primitive."""

    def __init__(self, guide_scale=1.75, group_size=2):
        if group_size < 1:
            raise ValueError('group_size must be positive')
        if not math.isfinite(guide_scale):
            raise ValueError('guide_scale must be finite')
        self.guide_scale = guide_scale
        self.group_size = group_size
