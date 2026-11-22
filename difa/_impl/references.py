# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import torch
import torch.nn.functional as F
from .base import RefinementBase, Refinement
from .consensus import ConsensusDiFA
from .components import ComponentDiFA
from .anchors import correction_from_anchor
from .views import instant_anchor, axis_geometry, off_axis
DIMS = (1, 2, 3)

def norm(value):
    return torch.linalg.vector_norm(value, dim=DIMS, keepdim=True)

def rms(value):
    return float(value.square().mean().sqrt())

def memory_reference(prediction, moments, mass):
    """DiFA prediction-refinement primitive."""
    if len(moments) != 2 or mass <= 0.0:
        raise ValueError('Expected two component moments and positive mass')
    low = F.avg_pool2d(prediction, 5, 1, 2)
    anchor = torch.zeros_like(prediction)
    for part, moment in zip((low, prediction - low), moments):
        if moment.shape != prediction.shape or not bool(torch.isfinite(moment).all()):
            raise ValueError('Malformed raw branch memory')
        mean = part.mean((-2, -1), keepdim=True)
        std = (part - mean).square().mean((-2, -1), keepdim=True).sqrt()
        anchor = anchor + mean + std * (moment.to(prediction) / mass)
    return anchor

class ReferenceDiFA(RefinementBase):

    def __init__(self, guide_scale=1.75, group_size=2):
        super().__init__(guide_scale, group_size)
        self.method = 'temporal_reference'
        self.parent = ConsensusDiFA(guide_scale, group_size)
        self.memory_axis = ComponentDiFA('logsnr_memory', guide_scale, group_size)

    @torch.no_grad()
    def refine(self, prediction, history=(), *, log_snr, high_frequency_weight, memory=None, mass=0.0, peers=None, peer_memory=None, peer_mass=0.0, mixing=None):
        parent = self.parent.refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass)
        ready = memory is not None and mass > 0.0
        info = dict(parent.diagnostics, reference_active_fraction=0.0, reference_vs_parent_RMS=0.0, reference_cold_fraction=float(not ready), reference_history_ready_fraction=float(ready), reference_delta_over_v02_norm_mean=0.0)
        if self.guide_scale <= 1.0 or peers is None or mixing is None:
            return Refinement(parent.correction, info)
        if len(peers) != 2:
            raise ValueError('Two raw data views are required')
        for peer in peers:
            if (peer.shape, peer.dtype, peer.device) != (prediction.shape, prediction.dtype, prediction.device):
                raise ValueError('Raw view metadata does not match prediction')
            if not bool(torch.isfinite(peer).all()):
                raise ValueError('Nonfinite raw view')
        if mixing.shape != (1, prediction.shape[1], 1, 1) or mixing.dtype != prediction.dtype or mixing.device != prediction.device or (not bool(torch.isfinite(mixing).all())):
            raise ValueError('Expected finite per-channel view mixing')
        dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
        d = prediction.to(dtype)
        contrast = (peers[0] - peers[1]).to(dtype)
        response_unit, valid = axis_geometry(prediction, contrast)
        if not bool(valid.any()):
            return Refinement(parent.correction, info)
        axis = self.memory_axis.refine(prediction, (), log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass).correction.to(dtype)
        unit, axis_valid = axis_geometry(prediction, axis)
        anchor = instant_anchor(prediction, peers)
        instant = correction_from_anchor(prediction, anchor, self.guide_scale, high_frequency_weight).to(dtype)
        project = lambda value: torch.where(axis_valid, off_axis(value, unit), value)
        h = project(instant)
        info.update(reference_v02_addition_RMS=rms(h), reference_raw_response_RMS=rms(contrast))
        delta = h
        if ready:
            if peer_memory is None or peer_mass <= 0.0:
                delta = torch.zeros_like(h)
                info['reference_peer_memory_missing'] = 1.0
            else:
                if len(peer_memory) != 2:
                    raise ValueError('Expected two independent branch memories')
                old_anchor = 0.5 * (memory_reference(d, peer_memory[0], peer_mass) + memory_reference(d, peer_memory[1], peer_mass))
                old_instant = correction_from_anchor(d, old_anchor, self.guide_scale, high_frequency_weight)
                delta = project(instant - old_instant)
                info['reference_reference_history_RMS'] = rms(old_instant)
        delta = torch.where(valid, delta, torch.zeros_like(delta))
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError('Nonfinite mechanism correction')
        active = valid & (norm(delta) > torch.finfo(prediction.dtype).eps * norm(d))
        proposal = parent.correction.to(dtype) + delta
        rounded = prediction + proposal.to(prediction.dtype) - prediction
        correction = torch.where(active, rounded, parent.correction)
        if not bool(torch.isfinite(correction).all()):
            raise RuntimeError('Nonfinite mechanism output')
        added = correction.to(dtype) - parent.correction.to(dtype)
        nh = norm(h)
        fraction = torch.where(nh > 0.0, norm(delta) / torch.where(nh > 0.0, nh, torch.ones_like(nh)), torch.zeros_like(nh))
        info.update(reference_active_fraction=float(active.float().mean()), reference_vs_parent_RMS=rms(added), reference_delta_over_v02_norm_mean=float(fraction.mean()), reference_response_projection_before=float((h * response_unit).sum(DIMS).mean()), reference_response_projection_after=float((added * response_unit).sum(DIMS).mean()), reference_response_projection_absmax=float((added * response_unit).sum(DIMS).abs().max()), reference_axis_preservation_error_RMS=rms(torch.where(axis_valid, (added * unit).sum(DIMS, keepdim=True), torch.zeros_like(added[:, :1, :1, :1]))))
        return Refinement(correction, info)
