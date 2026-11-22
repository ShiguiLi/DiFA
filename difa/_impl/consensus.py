# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import math
import torch
import torch.nn.functional as F
from .base import RefinementBase, Refinement
from .coordinates import CoordinateConsensusDiFA, coordinate_groups
from .guidance import GuidanceDiFA, _pair_geometry
from .components import ComponentDiFA, _aligned_values, binomial_filter
from .anchors import radial_residual
_DIMS = (1, 2, 3)

def _rms(value):
    dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    return float(value.to(dtype).square().mean().sqrt())

def project_off_axis(value, unit):
    """DiFA prediction-refinement primitive."""
    return value - (value * unit).sum(_DIMS, keepdim=True) * unit

def filtered_tangent(value, prediction, guide_scale, high_frequency_weight):
    """DiFA prediction-refinement primitive."""
    residual = radial_residual(value, prediction)
    low = binomial_filter(residual)
    return (guide_scale - 1.0) * (low + high_frequency_weight * (residual - low))

def duplicate_geometry(prediction, history, *, log_snr, guide_scale, high_frequency_weight, axis):
    """DiFA prediction-refinement primitive."""
    groups = sorted(coordinate_groups([record[1] for record in history]), key=lambda group: group['rho'])
    duplicates = [group for group in groups if len(group['indices']) > 1]
    if len(groups) < 2 or not duplicates:
        return None
    latest_duplicate = duplicates[-1]
    previous, latest = groups[-2:]
    gap = latest['rho'] - previous['rho']
    if gap <= 1e-08:
        return None
    a, _, d, unit, _, axis_valid, _, _, nd = _pair_geometry(axis, axis, prediction)
    old = torch.stack([record[0].to(d.dtype) for record in history], 1)
    batch, count, channels, height, width = old.shape
    current_low = F.avg_pool2d(d, 5, 1, 2)
    old_low = F.avg_pool2d(old.reshape(batch * count, channels, height, width), 5, 1, 2).reshape_as(old)
    aligned = (_aligned_values(current_low, old_low)[0], _aligned_values(d - current_low, old - old_low)[0])
    indices = latest_duplicate['indices']
    first, last = (indices[0], indices[-1])
    q_raw = sum((values[:, first] - values[:, last] for values in aligned)) / math.sqrt(2.0)
    signal_raw = sum((values[:, latest['indices']].mean(1) - values[:, previous['indices']].mean(1) for values in aligned))
    q = project_off_axis(filtered_tangent(q_raw, d, guide_scale, high_frequency_weight), unit)
    signal = project_off_axis(filtered_tangent(signal_raw, d, guide_scale, high_frequency_weight), unit)
    v = q.square().sum(_DIMS, keepdim=True)
    s = signal.square().sum(_DIMS, keepdim=True)
    if not all((bool(torch.isfinite(value).all()) for value in (q, signal, v, s))):
        raise RuntimeError('Nonfinite repeated-coordinate geometry')
    threshold = torch.finfo(prediction.dtype).eps * nd
    valid = axis_valid & (v.sqrt() > threshold) & (s.sqrt() > threshold)
    denominator = torch.where(valid, v + s, torch.ones_like(v))
    r = torch.where(valid, v / denominator, torch.zeros_like(v))
    rho = 1.0 / (1.0 + math.exp(-0.5 * float(log_snr))) if log_snr >= 0.0 else math.exp(0.5 * float(log_snr)) / (1.0 + math.exp(0.5 * float(log_snr)))
    advance = max(0.0, rho - latest['rho'])
    displacement = advance / gap * signal_raw
    return dict(q=q, signal=signal, r=r, valid=valid, unit=unit, axis=a, displacement=displacement, diagnostics=dict(consensus_unique_coordinates=len(groups), consensus_duplicate_count=len(indices), consensus_duplicate_excess=len(history) - len(groups), consensus_disagreement_RMS=_rms(q), consensus_signal_RMS=_rms(signal), consensus_disagreement_energy_mean=float(v.mean()), consensus_signal_energy_mean=float(s.mean()), consensus_duplicate_valid_fraction=float(valid.float().mean()), consensus_damping_mean=float(r.mean()), consensus_damping_min=float(r.min()), consensus_damping_max=float(r.max()), consensus_secant_gap=gap, consensus_secant_advance=advance))

class ConsensusDiFA(RefinementBase):
    """DiFA prediction-refinement primitive."""

    def __init__(self, guide_scale=1.75, group_size=2):
        super().__init__(guide_scale, group_size)
        self.method = 'consensus'
        self.parent = CoordinateConsensusDiFA(guide_scale, group_size)
        self.guidance = GuidanceDiFA(guide_scale, group_size)
        self.memory_axis = ComponentDiFA('logsnr_memory', guide_scale, group_size)
        self.static = ComponentDiFA('band_binomial', guide_scale, group_size)

    @torch.no_grad()
    def refine(self, prediction, history=(), *, log_snr, high_frequency_weight, memory=None, mass=0.0):
        parent = self.parent.refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass)
        info = dict(parent.diagnostics, consensus_active_fraction=0.0, consensus_removed_RMS=0.0, consensus_vs_d02_RMS=0.0, consensus_duplicate_valid_fraction=0.0, consensus_damping_mean=0.0)
        if not history or memory is None or mass <= 0.0 or (self.guide_scale <= 1.0):
            return Refinement(parent.correction, info)
        transport_active = bool(parent.diagnostics.get('guidance_transport_active', 0.0))
        axis = self.memory_axis.refine(prediction, (), log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass).correction
        geometry = duplicate_geometry(prediction, history, log_snr=log_snr, guide_scale=self.guide_scale, high_frequency_weight=high_frequency_weight, axis=axis)
        if geometry is None:
            return Refinement(parent.correction, info)
        info.update(geometry['diagnostics'])
        q, r, valid, unit = (geometry[key] for key in ('q', 'r', 'valid', 'unit'))
        dtype = q.dtype
        c0 = self.guidance.refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass).correction
        actual = parent.correction.to(dtype) - c0.to(dtype)
        v = q.square().sum(_DIMS, keepdim=True)
        coefficient = (actual * q).sum(_DIMS, keepdim=True) / torch.where(valid, v, torch.ones_like(v))
        removed = r * coefficient * q
        removed = torch.where(valid, removed, torch.zeros_like(removed))
        delta = -removed
        info['consensus_supplemental_static_RMS'] = _rms(actual)
        active = valid & (r > 0.0) & (delta.square().sum(_DIMS, keepdim=True) > 0.0)
        proposal = parent.correction.to(dtype) + delta
        rounded = prediction + proposal.to(prediction.dtype) - prediction
        correction = torch.where(active, rounded, parent.correction)
        if not bool(torch.isfinite(correction).all()):
            raise RuntimeError('Nonfinite next3 refinement')
        actual_delta = correction.to(dtype) - parent.correction.to(dtype)
        info.update(consensus_active_fraction=float(active.float().mean()), consensus_removed_RMS=_rms(removed), consensus_vs_d02_RMS=_rms(actual_delta), consensus_axis_preservation_error_RMS=_rms((actual_delta * unit).sum(_DIMS, keepdim=True)))
        return Refinement(correction, info)
