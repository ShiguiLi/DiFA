# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import math
import torch
import torch.nn.functional as F
from .base import RefinementBase, Refinement, _to_groups, _from_groups
REFERENCE_MODES = ('band_binomial', 'logsnr_memory')

def exponential_linear_weights(delta):
    """DiFA prediction-refinement primitive."""
    delta = float(delta)
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError('Expected a finite nonnegative log-SNR interval')
    if delta == 0.0:
        return (1.0, 0.0, 0.0)
    old = math.exp(-delta)
    mass = -math.expm1(-delta)
    if delta < 0.001:
        right = delta * (0.5 + delta * (-1.0 / 6.0 + delta * (1.0 / 24.0 + delta * (-1.0 / 120.0 + delta * (1.0 / 720.0 - delta / 5040.0)))))
        left = mass - right
    else:
        average = mass / delta
        left, right = (average - old, 1.0 - average)
    return (old, left, right)

def component_statistics(prediction):
    low = F.avg_pool2d(prediction, 5, 1, 2)
    components = (low, prediction - low)
    result = []
    for value in components:
        mean = value.mean((-2, -1), keepdim=True)
        std = (value - mean).square().mean((-2, -1), keepdim=True).sqrt()
        result.append((value, mean, std))
    return tuple(result)

def standardize_components(prediction):
    """DiFA prediction-refinement primitive."""
    return tuple(((value - mean) / std.clamp_min(1e-06) for value, mean, std in component_statistics(prediction)))

def _aligned_values(current, old):
    mean = current.mean((-2, -1), keepdim=True)
    std = (current - mean).square().mean((-2, -1), keepdim=True).sqrt()
    old_mean = old.mean((-2, -1), keepdim=True)
    old_std = (old - old_mean).square().mean((-2, -1), keepdim=True).sqrt()
    values = mean[:, None] + (old - old_mean) * std[:, None] / old_std.clamp_min(1e-06)
    return (values, float((old_std < 1e-06).float().mean()))

def binomial_filter(value):
    kernel = value.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    channels = value.shape[1]
    horizontal = kernel.reshape(1, 1, 1, 5).expand(channels, 1, 1, 5)
    vertical = kernel.reshape(1, 1, 5, 1).expand(channels, 1, 5, 1)
    value = F.conv2d(value, horizontal, padding=(0, 2), groups=channels)
    return F.conv2d(value, vertical, padding=(2, 0), groups=channels)

class ComponentDiFA(RefinementBase):

    def __init__(self, method, guide_scale=1.75, group_size=2):
        super().__init__(guide_scale, group_size)
        if method not in REFERENCE_MODES:
            raise ValueError('Unknown component reference mode: ' + str(method))
        self.method = method

    @torch.no_grad()
    def refine(self, prediction, history=(), *, log_snr, high_frequency_weight, memory=None, mass=0.0):
        if prediction.ndim != 4 or not prediction.is_floating_point():
            raise ValueError('Expected floating prediction[B,C,H,W]')
        height, width = prediction.shape[-2:]
        if height % self.group_size or width % self.group_size:
            raise ValueError('Spatial shape is not divisible by group size')
        if not math.isfinite(log_snr) or not 0.0 <= high_frequency_weight <= 1.0:
            raise ValueError('Invalid schedule values')
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError('Invalid memory mass')
        info = dict(high_frequency_gate=high_frequency_weight)
        use_memory = self.method == 'logsnr_memory'
        available = memory is not None and mass > 0.0 if use_memory else bool(history)
        if not available or self.guide_scale <= 1.0:
            return Refinement(torch.zeros_like(prediction), info)
        if use_memory:
            if len(memory) != 2:
                raise ValueError('Expected smooth and residual component memories')
            anchors = []
            for (value, mean, std), moment in zip(component_statistics(prediction), memory):
                if (moment.shape, moment.dtype, moment.device) != (prediction.shape, prediction.dtype, prediction.device):
                    raise ValueError('Memory must match prediction')
                anchors.append(mean + std * (moment / mass))
            anchor = anchors[0] + anchors[1]
            info['memory_mass'] = mass
        else:
            for value, ell, _ in history:
                if (value.shape, value.dtype, value.device) != (prediction.shape, prediction.dtype, prediction.device):
                    raise ValueError('History must match prediction')
                if not math.isfinite(ell):
                    raise ValueError('Historical log-SNR must be finite')
            old = torch.stack([record[0] for record in history], 1)
            logits = prediction.new_tensor([-abs(log_snr - record[1]) for record in history])
            weights = logits.softmax(0)[None].expand(len(prediction), -1)
            current_low = F.avg_pool2d(prediction, 5, 1, 2)
            batch, count, channels, h, w = old.shape
            old_low = F.avg_pool2d(old.reshape(batch * count, channels, h, w), 5, 1, 2).reshape_as(old)
            low_values, low_degenerate = _aligned_values(current_low, old_low)
            high_values, high_degenerate = _aligned_values(prediction - current_low, old - old_low)
            aligned = low_values + high_values
            values = _to_groups(aligned, self.group_size)
            grouped_anchor = (weights[:, :, None, None] * values).sum(1)
            anchor = _from_groups(grouped_anchor, height, width, self.group_size)
            info.update(history_count=len(history), low_alignment_degenerate_fraction=low_degenerate, residual_alignment_degenerate_fraction=high_degenerate, weight_entropy=float(-(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(1).mean()), latest_weight_mean=float(weights[:, -1].mean()))
        deviation = prediction - anchor
        dims = tuple(range(1, prediction.ndim))
        coefficient = (deviation * prediction).sum(dims, keepdim=True) / (prediction.square().sum(dims, keepdim=True) + 1e-08)
        residual = deviation - coefficient * prediction
        low = binomial_filter(residual)
        correction = (self.guide_scale - 1.0) * (low + high_frequency_weight * (residual - low))
        correction = prediction + correction - prediction
        return Refinement(correction, info)
