# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
import math
import torch
from .base import RefinementBase, Refinement
from .components import ComponentDiFA, exponential_linear_weights
from .anchors import build_anchors, correction_from_anchor
_DIMS = (1, 2, 3)

def _norm(value):
    return torch.linalg.vector_norm(value, dim=_DIMS, keepdim=True)

def _dot(left, right):
    return (left * right).sum(_DIMS, keepdim=True)

def _rms(value):
    dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    return float(value.to(dtype).square().mean().sqrt())

class LogSNRMemory:
    """DiFA prediction-refinement primitive."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.moments = None
        self.mass = 0.0
        self.last_level = None
        self.last_shapes = None
        self.intervals = 0
        self.observations = 0

    def snapshot(self):
        return (self.moments, self.mass, self.last_level, self.last_shapes, self.intervals, self.observations)

    def restore(self, state):
        self.moments, self.mass, self.last_level, self.last_shapes, self.intervals, self.observations = state

    @torch.no_grad()
    def commit(self, level, shapes):
        level = float(level)
        if not math.isfinite(level) or len(shapes) != 2:
            raise ValueError('Expected finite log-SNR and two standardized components')
        if not all((value.ndim == 4 and value.is_floating_point() and bool(torch.isfinite(value).all()) for value in shapes)):
            raise ValueError('Memory observations must be finite floating BCHW tensors')
        if shapes[0].shape != shapes[1].shape:
            raise ValueError('Memory components must have matching shapes')
        if self.last_shapes is not None:
            for value, previous in zip(shapes, self.last_shapes):
                if (value.shape, value.dtype, value.device) != (previous.shape, previous.dtype, previous.device):
                    raise ValueError('Memory observation shape, dtype or device changed')
        if self.last_level is not None and level < self.last_level:
            raise ValueError('Memory requires nondecreasing committed log-SNR; restore before retry')
        moments, mass = (self.moments, self.mass)
        integrated = self.last_level is not None and level > self.last_level
        if integrated:
            old, left, right = exponential_linear_weights(level - self.last_level)
            observed = tuple((left * z0 + right * z1 for z0, z1 in zip(self.last_shapes, shapes)))
            moments = observed if moments is None else tuple((old * previous + value for previous, value in zip(moments, observed)))
            mass = old * mass + left + right
            if not all((bool(torch.isfinite(value).all()) for value in moments)):
                raise RuntimeError('Nonfinite memory update; state was not changed')
        saved_shapes = tuple((value.detach().clone() for value in shapes))
        self.moments, self.mass = (moments, mass)
        self.last_level, self.last_shapes = (level, saved_shapes)
        self.intervals += int(integrated)
        self.observations += 1

def _pair_geometry(axis, transported, prediction):
    """DiFA prediction-refinement primitive."""
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    a, t, d = (value.to(dtype) for value in (axis, transported, prediction))
    na, nt, nd = (_norm(a), _norm(t), _norm(d))
    if not all((bool(torch.isfinite(value).all()) for value in (a, t, d, na, nt, nd))):
        raise RuntimeError('Nonfinite guidance geometry')
    threshold = torch.finfo(prediction.dtype).eps * nd
    valid = (na > threshold) & (nt > threshold) & (na > 0.0) & (nt > 0.0)
    safe = torch.where(valid, na, torch.ones_like(na))
    unit = a / safe
    innovation = t - _dot(t, unit) * unit
    innovation = torch.where(valid, innovation, torch.zeros_like(innovation))
    return (a, t, d, unit, innovation, valid, na, nt, nd)

class GuidanceDiFA(RefinementBase):

    def __init__(self, guide_scale=1.75, group_size=2):
        super().__init__(guide_scale, group_size)
        self.method = 'memory_axis'
        self.static = ComponentDiFA('band_binomial', guide_scale, group_size)
        self.memory_base = ComponentDiFA('logsnr_memory', guide_scale, group_size)

    @torch.no_grad()
    def refine(self, prediction, history=(), *, log_snr, high_frequency_weight, memory=None, mass=0.0):
        if prediction.ndim != 4 or not prediction.is_floating_point():
            raise ValueError('Expected floating prediction[B,C,H,W]')
        if min(prediction.shape) < 1:
            raise ValueError('Prediction dimensions must be nonempty')
        if any((size % self.group_size for size in prediction.shape[-2:])):
            raise ValueError('Spatial shape must be divisible by group_size')
        if not math.isfinite(log_snr) or not 0.0 <= high_frequency_weight <= 1.0:
            raise ValueError('Invalid schedule values')
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError('Invalid memory mass')
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError('Prediction must be finite')
        for record in history:
            value, level = record[:2]
            if (value.shape, value.dtype, value.device) != (prediction.shape, prediction.dtype, prediction.device):
                raise ValueError('History must match prediction shape, dtype and device')
            if not math.isfinite(level) or not bool(torch.isfinite(value).all()):
                raise ValueError('Historical prediction/log-SNR must be finite')
        memory_ready = memory is not None and mass > 0.0
        if memory is not None:
            if len(memory) != 2:
                raise ValueError('Expected two component memory moments')
            for value in memory:
                if (value.shape, value.dtype, value.device) != (prediction.shape, prediction.dtype, prediction.device):
                    raise ValueError('Memory must match prediction shape, dtype and device')
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('Memory moments must be finite')
        if memory_ready:
            base = self.memory_base.refine(prediction, (), log_snr=log_snr, high_frequency_weight=high_frequency_weight, memory=memory, mass=mass)
        else:
            base = self.static.refine(prediction, history, log_snr=log_snr, high_frequency_weight=high_frequency_weight)
        info = dict(base.diagnostics)
        info.update(history_count=len(history), guidance_transport_active=0.0, guidance_pair_valid_fraction=0.0, guidance_fallback_fraction=0.0)
        info.update(guidance_memory_ready=float(memory_ready), guidance_memory_fallback=float(not memory_ready), guidance_memory_mass=mass)
        if not memory_ready:
            return Refinement(base.correction, info)
        if not history or self.guide_scale <= 1.0:
            return Refinement(base.correction, info)
        components, static, predicted, transported, eta, extra = build_anchors(prediction, history, log_snr=log_snr, group_size=self.group_size)
        info.update(extra)
        info['guidance_transport_active'] = float(eta > 0.0)
        if eta == 0.0:
            return Refinement(base.correction, info)
        if not all((bool(torch.isfinite(value).all()) for value in static + predicted + transported)):
            raise RuntimeError('Nonfinite transported anchor')
        trans = correction_from_anchor(prediction, transported[0] + transported[1], self.guide_scale, high_frequency_weight)
        a, t, d, unit, innovation, valid, na, nt, nd = _pair_geometry(base.correction, trans, prediction)
        info.update(guidance_axis_RMS=_rms(a), guidance_transport_RMS=_rms(t), guidance_pair_valid_fraction=float(valid.float().mean()), guidance_fallback_fraction=float((~valid).float().mean()))
        safe_a = torch.where(valid, na, torch.ones_like(na))
        safe_t = torch.where(valid, nt, torch.ones_like(nt))
        cosine = _dot(a / safe_a, t / safe_t)
        info['guidance_pair_cosine_sum_per_sample'] = float(cosine[valid].sum()) / len(prediction)
        info['guidance_transport_over_axis_norm_sum_per_sample'] = float((nt / safe_a)[valid].sum()) / len(prediction)
        proposal = a + innovation
        info['guidance_innovation_RMS'] = _rms(innovation)
        rounded = prediction + proposal.to(prediction.dtype) - prediction
        if not bool(torch.isfinite(rounded).all()):
            raise RuntimeError('Nonfinite guidance4 correction')
        info['guidance_output_RMS'] = _rms(rounded)
        return Refinement(rounded, info)
