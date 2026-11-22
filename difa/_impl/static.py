# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import torch
from .base import RefinementBase, Refinement
from .guidance import GuidanceDiFA, _pair_geometry
from .components import ComponentDiFA
from .anchors import build_anchors, correction_from_anchor
_DIMS = (1, 2, 3)

class StaticDiFA(RefinementBase):
    """DiFA prediction-refinement primitive."""

    def __init__(self, guide_scale=1.75, group_size=2):
        super().__init__(guide_scale, group_size)
        self.parent = GuidanceDiFA(guide_scale, group_size)
        self.memory_axis = ComponentDiFA('logsnr_memory', guide_scale, group_size)
        self.static = ComponentDiFA('band_binomial', guide_scale, group_size)

    @torch.no_grad()
    def refine(self, prediction, history=(), *, log_snr, high_frequency_weight, memory=None, mass=0.0):
        parent = self.parent.refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass)
        info = dict(parent.diagnostics, static_always_active=0.0)
        if not history or self.guide_scale <= 1.0 or memory is None or (mass <= 0.0):
            return Refinement(parent.correction, info)
        axis = self.memory_axis.refine(prediction, (), log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass).correction
        static = self.static.refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight).correction
        transport_active = bool(parent.diagnostics.get('guidance_transport_active', 0.0))
        if transport_active:
            _, _, _, transported, eta, _ = build_anchors(prediction, history, log_snr=log_snr, group_size=self.group_size)
            trans = correction_from_anchor(prediction, transported[0] + transported[1], self.guide_scale, high_frequency_weight)
        else:
            eta, trans = (0.0, axis)
        a, _, _, unit, _, valid, _, _, _ = _pair_geometry(axis, trans, prediction)
        static = static.to(a.dtype)
        x = static - (static * unit).sum(_DIMS, keepdim=True) * unit
        x = torch.where(valid, x, torch.zeros_like(x))
        rounded = prediction + (parent.correction.to(a.dtype) + x).to(prediction.dtype) - prediction
        correction = torch.where(valid, rounded, parent.correction)
        if not bool(torch.isfinite(correction).all()):
            raise RuntimeError('Nonfinite static-always refinement')
        info.update(static_always_active=float(valid.float().mean()), static_always_valid_fraction=float(valid.float().mean()), static_always_transport_eta=eta, static_always_new_region_fraction=float(valid.float().mean()) if not transport_active else 0.0, static_increment_RMS=float(x.square().mean().sqrt()), static_applied_increment_RMS=float((correction.to(a.dtype) - parent.correction.to(a.dtype)).square().mean().sqrt()), static_always_output_RMS=float(correction.square().mean().sqrt()))
        return Refinement(correction, info)
