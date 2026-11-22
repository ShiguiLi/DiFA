# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import math
import torch
from .base import Refinement
from .references import ReferenceDiFA, memory_reference, norm, rms
from .components import component_statistics, standardize_components
from .anchors import correction_from_anchor
from .views import instant_anchor, axis_geometry, off_axis

class BranchInnovationDiFA(ReferenceDiFA):

    def __init__(self, guide_scale=1.75, group_size=2):
        super().__init__(guide_scale, group_size)
        self.variant = 'branch_innovation'

    @torch.no_grad()
    def refine(self, prediction, history=(), *, log_snr, high_frequency_weight, memory=None, mass=0.0, peers=None, peer_memory=None, peer_mass=0.0, mixing=None):
        ready = memory is not None and mass > 0.0
        inherited = dict(log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass, peers=peers, peer_memory=peer_memory, peer_mass=peer_mass, mixing=mixing)
        if not ready or self.guide_scale <= 1.0 or peers is None or (mixing is None) or (peer_memory is None) or (peer_mass <= 0.0):
            baseline = super().refine(prediction, history, **inherited)
            info = dict(baseline.diagnostics, branch_active_fraction=baseline.diagnostics.get('reference_active_fraction', 0.0), branch_vs_temporal_reference_RMS=0.0, branch_exact_temporal_reference_fallback=1.0, branch_changed_from_temporal_reference_fraction=0.0, branch_cold_fraction=float(not ready))
            return Refinement(baseline.correction, info)
        if not math.isfinite(peer_mass):
            raise ValueError('Expected finite raw branch-memory mass')
        if len(peers) != 2 or len(peer_memory) != 2:
            raise ValueError('Expected two raw views and two branch memories')
        for value in peers:
            if (value.shape, value.dtype, value.device) != (prediction.shape, prediction.dtype, prediction.device):
                raise ValueError('Raw view metadata differs from prediction')
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite raw view')
        for moments in peer_memory:
            if len(moments) != 2:
                raise ValueError('Expected low/residual branch-memory components')
            for value in moments:
                if (value.shape, value.dtype, value.device) != (prediction.shape, prediction.dtype, prediction.device):
                    raise ValueError('Branch memory metadata differs from prediction')
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('Nonfinite branch memory')
        if mixing.shape != (1, prediction.shape[1], 1, 1) or mixing.dtype != prediction.dtype or mixing.device != prediction.device or (not bool(torch.isfinite(mixing).all())):
            raise ValueError('Expected finite per-channel view mixing')
        base = self.parent.refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass).correction
        dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
        d = prediction.to(dtype)
        branch = tuple((value.to(dtype) for value in peers))
        response = branch[0] - branch[1]
        response_unit, valid = axis_geometry(prediction, response)
        if not bool(valid.any()):
            return Refinement(base, dict(branch_active_fraction=0.0, branch_vs_temporal_reference_RMS=0.0, branch_exact_temporal_reference_fallback=1.0, branch_cold_fraction=0.0))
        axis = self.memory_axis.refine(prediction, (), log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass).correction.to(dtype)
        unit, axis_valid = axis_geometry(prediction, axis)
        project = lambda value: torch.where(axis_valid, off_axis(value, unit), value)
        current_reference = instant_anchor(prediction, peers).to(dtype)
        past_reference = 0.5 * (memory_reference(d, peer_memory[0], peer_mass) + memory_reference(d, peer_memory[1], peer_mass))
        G = lambda anchor: correction_from_anchor(d, anchor, self.guide_scale, high_frequency_weight).to(dtype)
        temporal_delta = project(G(current_reference) - G(past_reference))
        stats = component_statistics(d)
        current_shapes = tuple((standardize_components(value) for value in branch))
        past_shapes = tuple((tuple((value.to(dtype) / peer_mass for value in moments)) for moments in peer_memory))
        current_parts = [mean + std * 0.5 * (current_shapes[0][i] + current_shapes[1][i]) for i, (_, mean, std) in enumerate(stats)]
        past_parts = [mean + std * 0.5 * (past_shapes[0][i] + past_shapes[1][i]) for i, (_, mean, std) in enumerate(stats)]
        extra = {}
        old_response = memory_reference(branch[0], peer_memory[0], peer_mass) - memory_reference(branch[1], peer_memory[1], peer_mass)
        innovation = response - old_response
        delta = (self.guide_scale - 1.0) * (mixing.to(dtype) - 0.5) * innovation
        extra.update(branch_conditional_memory_response_RMS=rms(old_response), branch_conditional_innovation_RMS=rms(innovation))
        delta = torch.where(valid, delta, torch.zeros_like(delta))
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError('Nonfinite reference innovation')

        def compose(value):
            active = valid & (norm(value) > torch.finfo(prediction.dtype).eps * norm(d))
            rounded = prediction + (base.to(dtype) + value).to(prediction.dtype) - prediction
            return (torch.where(active, rounded, base), active)
        correction, active = compose(delta)
        shadow, _ = compose(temporal_delta)
        if not bool(torch.isfinite(correction).all()):
            raise RuntimeError('Nonfinite innovation output')
        added = correction.to(dtype) - base.to(dtype)
        info = dict(branch_active_fraction=float(active.float().mean()), branch_changed_from_temporal_reference_fraction=float((norm(correction.to(dtype) - shadow.to(dtype)) > torch.finfo(prediction.dtype).eps * norm(d)).float().mean()), branch_exact_temporal_reference_fallback=0.0, branch_cold_fraction=0.0, branch_vs_temporal_reference_RMS=rms(correction.to(dtype) - shadow.to(dtype)), branch_temporal_reference_addition_RMS=rms(shadow.to(dtype) - base.to(dtype)), branch_addition_RMS=rms(added), branch_output_RMS=rms(correction.to(dtype)), branch_current_reference_RMS=rms(current_reference), branch_past_reference_RMS=rms(past_reference), branch_response_projection_after=float((added * response_unit).sum((1, 2, 3)).mean()), branch_axis_constrained=0.0, branch_axis_projection_RMS=rms(torch.where(axis_valid, (added * unit).sum((1, 2, 3), keepdim=True), torch.zeros_like(added[:, :1, :1, :1]))))
        info.update(extra)
        return Refinement(correction, info)
