# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import torch
import torch.nn.functional as F
from .components import _aligned_values
_DIMS = (1, 2, 3)

def _norm(value):
    return torch.linalg.vector_norm(value, dim=_DIMS, keepdim=True)

def axis_geometry(prediction, axis):
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    a, d = (axis.to(dtype), prediction.to(dtype))
    na, nd = (_norm(a), _norm(d))
    valid = (na > torch.finfo(prediction.dtype).eps * nd) & (na > 0.0)
    unit = a / torch.where(valid, na, torch.ones_like(na))
    return (unit, valid)

def off_axis(value, unit):
    return value - (value * unit).sum(_DIMS, keepdim=True) * unit

def instant_anchor(prediction, peers):
    """DiFA prediction-refinement primitive."""
    old = torch.stack(peers, dim=1)
    batch, count, channels, height, width = old.shape
    current_low = F.avg_pool2d(prediction, 5, 1, 2)
    old_low = F.avg_pool2d(old.reshape(batch * count, channels, height, width), 5, 1, 2).reshape_as(old)
    aligned_low, _ = _aligned_values(current_low, old_low)
    aligned_high, _ = _aligned_values(prediction - current_low, old - old_low)
    return (aligned_low + aligned_high).mean(1)
