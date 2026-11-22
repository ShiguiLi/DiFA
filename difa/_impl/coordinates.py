# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import torch
import torch.nn.functional as F
from .base import Refinement, _to_groups, _from_groups
from .static import StaticDiFA
from .guidance import _pair_geometry
from .components import _aligned_values
from .anchors import correction_from_anchor, progress_from_log_snr
COORDINATE_TOLERANCE = 2e-07
_DIMS = (1, 2, 3)

def coordinate_groups(log_snrs, tolerance=COORDINATE_TOLERANCE):
    """DiFA prediction-refinement primitive."""
    groups = []
    for index, level in enumerate(log_snrs):
        rho = progress_from_log_snr(level)
        match = next((group for group in groups if abs(group['rho'] - rho) <= tolerance), None)
        if match is None:
            groups.append(dict(rho=rho, ell=[float(level)], indices=[index]))
        else:
            match['ell'].append(float(level))
            match['indices'].append(index)
    return groups

def coordinate_balanced_static(prediction, history, log_snr, group_size):
    """DiFA prediction-refinement primitive."""
    old = torch.stack([record[0] for record in history], 1)
    channels, height, width = old.shape[2:]
    current_low = F.avg_pool2d(prediction, 5, 1, 2)
    old_low = F.avg_pool2d(old.reshape(-1, channels, height, width), 5, 1, 2)
    old_low = old_low.reshape_as(old)
    components = (_aligned_values(current_low, old_low)[0], _aligned_values(prediction - current_low, old - old_low)[0])
    levels = [float(record[1]) for record in history]
    groups = coordinate_groups(levels)
    aligned_by_coordinate = []
    group_levels = []
    for group in groups:
        indices = group['indices']
        aligned_by_coordinate.append(sum((value[:, indices].mean(1) for value in components)))
        group_levels.append(sum(group['ell']) / len(group['ell']))
    coordinate_values = torch.stack(aligned_by_coordinate, 1)
    logits = prediction.new_tensor([-abs(log_snr - level) for level in group_levels])
    weights = logits.softmax(0)[None].expand(len(prediction), -1)
    grouped = _to_groups(coordinate_values, group_size)
    anchor = _from_groups((weights[:, :, None, None] * grouped).sum(1), height, width, group_size)
    return (anchor, dict(coordinate_unique_levels=len(groups), coordinate_duplicate_excess=len(history) - len(groups)))

class CoordinateConsensusDiFA(StaticDiFA):
    """DiFA prediction-refinement primitive."""

    @torch.no_grad()
    def refine(self, prediction, history=(), *, log_snr, high_frequency_weight, memory=None, mass=0.0):
        parent = super().refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass)
        groups = coordinate_groups([record[1] for record in history]) if history else []
        info = dict(parent.diagnostics, coordinate_unique_levels=len(groups), coordinate_duplicate_excess=len(history) - len(groups), coordinate_balanced_active=0.0, coordinate_reference_RMS=0.0)
        if len(groups) == len(history) or self.guide_scale <= 1.0 or memory is None or (mass <= 0.0) or bool(parent.diagnostics.get('guidance_transport_active', 0.0)):
            return Refinement(parent.correction, info)
        axis = self.memory_axis.refine(prediction, (), log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass).correction
        anchor, coord_info = coordinate_balanced_static(prediction, history, log_snr, self.group_size)
        coordinate_static = correction_from_anchor(prediction, anchor, self.guide_scale, high_frequency_weight)
        a, _, _, unit, _, valid, _, _, _ = _pair_geometry(axis, axis, prediction)
        x = coordinate_static.to(a.dtype)
        x = x - (x * unit).sum(_DIMS, keepdim=True) * unit
        x = torch.where(valid, x, torch.zeros_like(x))
        rounded = prediction + (a + x).to(prediction.dtype) - prediction
        correction = torch.where(valid, rounded, axis)
        if not bool(torch.isfinite(correction).all()):
            raise RuntimeError('Nonfinite coordinate-balanced refinement')
        info.update(coord_info, coordinate_balanced_active=float(valid.float().mean()), coordinate_reference_RMS=float((correction.to(a.dtype) - parent.correction.to(a.dtype)).square().mean().sqrt()), static_always_active=float(valid.float().mean()), static_always_valid_fraction=float(valid.float().mean()), static_always_new_region_fraction=float(valid.float().mean()), static_increment_RMS=float(x.square().mean().sqrt()), static_applied_increment_RMS=float((correction.to(a.dtype) - a).square().mean().sqrt()), static_always_output_RMS=float(correction.square().mean().sqrt()))
        return Refinement(correction, info)
