# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import math
import torch
import torch.nn.functional as F
from .base import _to_groups, _from_groups
from .components import _aligned_values, binomial_filter

def progress_from_log_snr(log_snr):
    """DiFA prediction-refinement primitive."""
    level = 0.5 * float(log_snr)
    if not math.isfinite(level):
        raise ValueError('log_snr must be finite')
    if level >= 0.0:
        return 1.0 / (1.0 + math.exp(-level))
    value = math.exp(level)
    return value / (1.0 + value)

def radial_residual(value, prediction):
    """DiFA prediction-refinement primitive."""
    dims = tuple(range(1, prediction.ndim))
    coefficient = (value * prediction).sum(dims, keepdim=True) / (prediction.square().sum(dims, keepdim=True) + 1e-08)
    return value - coefficient * prediction

def correction_from_anchor(prediction, anchor, guide_scale, high_frequency_weight):
    residual = radial_residual(prediction - anchor, prediction)
    low = binomial_filter(residual)
    correction = (guide_scale - 1.0) * (low + high_frequency_weight * (residual - low))
    return prediction + correction - prediction

def _last_secant(aligned, coordinates):
    """DiFA prediction-refinement primitive."""
    groups = []
    for index, coordinate in enumerate(coordinates):
        found = next((group for group in groups if abs(group[0] - coordinate) <= 2e-07), None)
        if found is None:
            groups.append([coordinate, [index]])
        else:
            found[1].append(index)
    groups.sort(key=lambda group: group[0])
    if len(groups) < 2:
        return (None, None, None, None, len(groups))
    left_coordinate, left_indices = groups[-2]
    right_coordinate, right_indices = groups[-1]
    gap = right_coordinate - left_coordinate
    if gap <= 1e-08:
        return (None, None, None, None, len(groups))
    left = aligned[:, left_indices].mean(1)
    right = aligned[:, right_indices].mean(1)
    return ((right - left) / gap, right, right_coordinate, gap, len(groups))

def build_anchors(prediction, history, *, log_snr, group_size):
    """DiFA prediction-refinement primitive."""
    old = torch.stack([record[0] for record in history], 1)
    logits = prediction.new_tensor([-abs(log_snr - record[1]) for record in history])
    weights = logits.softmax(0)[None].expand(len(prediction), -1)
    current_low = F.avg_pool2d(prediction, 5, 1, 2)
    components = (current_low, prediction - current_low)
    batch, count, channels, height, width = old.shape
    old_low = F.avg_pool2d(old.reshape(batch * count, channels, height, width), 5, 1, 2).reshape_as(old)
    aligned = [_aligned_values(components[0], old_low), _aligned_values(components[1], old - old_low)]
    coordinates = [progress_from_log_snr(record[1]) for record in history]
    rho = progress_from_log_snr(log_snr)
    static, predicted, transported = ([], [], [])
    eta = 0.0
    unique_count = 0
    for values, _ in aligned:
        grouped = _to_groups(values, group_size)
        anchor = _from_groups((weights[:, :, None, None] * grouped).sum(1), height, width, group_size)
        slope, latest, last_rho, gap, unique_count = _last_secant(values, coordinates)
        static.append(anchor)
        if slope is None or rho <= last_rho:
            predicted.append(anchor)
            transported.append(anchor)
        else:
            advance = rho - last_rho
            extrapolated = latest + advance * slope
            eta = min(0.5, advance / (advance + gap + 1e-08))
            predicted.append(extrapolated)
            transported.append((1.0 - eta) * anchor + eta * extrapolated)
    diagnostics = dict(universal_progress=rho, universal_unique_progress_count=unique_count, universal_transport_eta=eta, universal_transport_active=float(eta > 0.0), low_alignment_degenerate_fraction=aligned[0][1], residual_alignment_degenerate_fraction=aligned[1][1], latest_weight_mean=float(weights[:, -1].mean()))
    return (components, static, predicted, transported, eta, diagnostics)
