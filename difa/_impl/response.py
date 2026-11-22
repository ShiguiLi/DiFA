# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import math
import torch
from .base import Refinement
from .innovation import BranchInnovationDiFA
from .references import memory_reference, norm, rms
from .components import component_statistics
from .views import axis_geometry

def response_decomposition(response, raw_memory, mass):
    """DiFA prediction-refinement primitive."""
    raw, shape, statistic = ([], [], [])
    small = []
    for (current, mu_c, std_c), moment in zip(component_statistics(response), raw_memory):
        past = moment / mass
        mu_m = past.mean((-2, -1), keepdim=True)
        std_m = (past - mu_m).square().mean((-2, -1), keepdim=True).sqrt()
        s_c, s_m = (std_c.clamp_min(1e-06), std_m.clamp_min(1e-06))
        z_c, z_m = ((current - mu_c) / s_c, (past - mu_m) / s_m)
        raw.append(current - past)
        shape.append(0.5 * (s_c + s_m) * (z_c - z_m))
        statistic.append(mu_c - mu_m + 0.5 * (s_c - s_m) * (z_c + z_m))
        small.append(((std_c < 1e-06) | (std_m < 1e-06)).float().mean())
    return (sum(raw), sum(shape), sum(statistic), float(torch.stack(small).mean()))

class DiFA(BranchInnovationDiFA):
    """Refine spatial clean-data estimates independently of model and solver.

    Inputs use BCHW data-prediction coordinates. Optional raw CFG views and
    synchronized component memories enable response-statistic refinement.
    The adapter applies the returned correction gain and commits history.
    """

    def __init__(self, guide_scale=1.75, group_size=2):
        super().__init__(guide_scale, group_size)
        self.response_variant = 'difa'

    @torch.no_grad()
    def refine(self, prediction, history=(), *, log_snr, high_frequency_weight, memory=None, mass=0.0, peers=None, peer_memory=None, peer_mass=0.0, mixing=None, response_memory=None, response_mass=0.0):
        """Return a correction direction and diagnostics without updating memory.

        The adapter supplies finite BCHW data estimates, log-SNR, a residual
        weight in [0, 1], and optional matching raw views/component memories.
        Missing views use consensus; incomplete memories use the reference
        fallback. This method performs no model evaluations or solver steps.
        """
        if not math.isfinite(response_mass) or response_mass < 0.0:
            raise ValueError('Expected finite nonnegative raw-response memory mass')
        inherited = dict(log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass, peers=peers, peer_memory=peer_memory, peer_mass=peer_mass, mixing=mixing)
        ready = memory is not None and mass > 0.0
        if not ready or self.guide_scale <= 1.0 or peers is None or (mixing is None) or (peer_memory is None) or (peer_mass <= 0.0) or (response_memory is None) or (response_mass <= 0.0):
            baseline = super().refine(prediction, history, **inherited)
            return Refinement(baseline.correction, dict(baseline.diagnostics, difa_exact_branch_fallback=1.0, difa_vs_branch_RMS=0.0, difa_active_fraction=baseline.diagnostics.get('branch_active_fraction', 0.0), difa_cold_fraction=float(not ready)))
        parent = self.parent.refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass)
        if len(peers) != 2 or len(peer_memory) != 2 or len(response_memory) != 2:
            raise ValueError('Expected two branches and two components per raw memory')
        if not math.isfinite(peer_mass):
            raise ValueError('Nonfinite raw branch-memory mass')
        for value in (*peers, *response_memory, *peer_memory[0], *peer_memory[1]):
            if (value.shape, value.dtype, value.device) != (prediction.shape, prediction.dtype, prediction.device):
                raise ValueError('Raw response/view/memory metadata mismatch')
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite raw response/view/memory')
        if any((len(moments) != 2 for moments in peer_memory)):
            raise ValueError('Expected two raw branch-memory components')
        if mixing.shape != (1, prediction.shape[1], 1, 1) or mixing.dtype != prediction.dtype or mixing.device != prediction.device or (not bool(torch.isfinite(mixing).all())):
            raise ValueError('Expected finite per-channel view mixing')
        dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
        d = prediction.to(dtype)
        branches = tuple((value.to(dtype) for value in peers))
        response = branches[0] - branches[1]
        response_unit, valid = axis_geometry(prediction, response)
        if not bool(valid.any()):
            return Refinement(parent.correction, dict(parent.diagnostics, difa_active_fraction=0.0, difa_exact_branch_fallback=1.0, difa_vs_branch_RMS=0.0))
        moments = tuple((value.to(dtype) for value in response_memory))
        full, shape, statistic, small = response_decomposition(response, moments, response_mass)
        old_branch_response = memory_reference(branches[0], peer_memory[0], peer_mass) - memory_reference(branches[1], peer_memory[1], peer_mass)
        branch_innovation = response - old_branch_response
        innovation = statistic
        gain = (self.guide_scale - 1.0) * (mixing.to(dtype) - 0.5)
        delta = torch.where(valid, gain * innovation, torch.zeros_like(innovation))
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError('Nonfinite response innovation')

        def compose(value):
            active = valid & (norm(value) > torch.finfo(prediction.dtype).eps * norm(d))
            rounded = prediction + (parent.correction.to(dtype) + value).to(prediction.dtype) - prediction
            return (torch.where(active, rounded, parent.correction), active)
        correction, active = compose(delta)
        shadow, _ = compose(torch.where(valid, gain * branch_innovation, torch.zeros_like(branch_innovation)))
        if not bool(torch.isfinite(correction).all()):
            raise RuntimeError('Nonfinite response refinement')
        added = correction.to(dtype) - parent.correction.to(dtype)
        info = dict(parent.diagnostics, difa_active_fraction=float(active.float().mean()), difa_exact_branch_fallback=0.0, difa_cold_fraction=0.0, difa_vs_branch_RMS=rms(correction.to(dtype) - shadow.to(dtype)), difa_addition_RMS=rms(added), difa_branch_addition_RMS=rms(shadow.to(dtype) - parent.correction.to(dtype)), difa_output_RMS=rms(correction.to(dtype)), difa_axis_constrained=0.0, difa_full_innovation_RMS=rms(full), difa_shape_innovation_RMS=rms(shape), difa_statistic_innovation_RMS=rms(statistic), difa_branch_innovation_RMS=rms(branch_innovation), difa_decomposition_error_RMS=rms(full - shape - statistic), difa_degenerate_scale_fraction=small, difa_raw_memory_RMS=rms(sum(moments) / response_mass), difa_response_projection_after=float((added * response_unit).sum((1, 2, 3)).mean()))
        return Refinement(correction, info)
