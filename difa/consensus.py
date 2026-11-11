"""Historical DiFA spatial consensus, independent of the prediction model."""
import math
import collections
from typing import Optional,Tuple
import torch
import torch.nn.functional as F

class DiFA_FM:
    """Historical DiFA consensus and orthogonal/spatial guidance.

This implementation is retained for the measured G03 pipeline. Historical
quality claims are not part of its numerical contract; see the root README."""

    def __init__(self, guide_scale=1.7, window_size=3, rescale_phi=1.0, channel_dim=1, spatial_shape: Optional[Tuple[int, ...]]=None):
        self.guide_scale = guide_scale
        self.window_size = max(1, int(window_size))
        self.rescale_phi = rescale_phi
        self.channel_dim = channel_dim
        self.spatial_shape = spatial_shape
        self._history_buffer = None

    def reset(self):
        self._history_buffer = None

    def _log_snr(self, t):
        t = max(min(float(t), 1.0 - 1e-06), 1e-06)
        return math.log(((1.0 - t) / t) ** 2 + 1e-08)

    def _to_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_dim == 1 or self.spatial_shape is None:
            return x
        B, C = (x.shape[0], x.shape[self.channel_dim])
        return x.transpose(1, self.channel_dim).reshape(B, C, *self.spatial_shape)

    def _from_spatial(self, x: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
        if self.channel_dim == 1 or self.spatial_shape is None:
            return x
        B, C = x.shape[:2]
        return x.reshape(B, C, -1).transpose(1, self.channel_dim)

    def _smooth(self, x: torch.Tensor, kernel_size=3) -> torch.Tensor:
        pad = kernel_size // 2
        if x.dim() == 4:
            return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        elif x.dim() == 5:
            B, C, T, H, W = x.shape
            x_s = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x_s = F.avg_pool2d(x_s, kernel_size=kernel_size, stride=1, padding=pad)
            x_s = x_s.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            x_t = x_s.permute(0, 1, 3, 4, 2).reshape(B * C * H * W, 1, T)
            x_t = F.avg_pool1d(x_t, kernel_size=3, stride=1, padding=1)
            return x_t.reshape(B, C, H, W, T).permute(0, 1, 4, 2, 3)
        return x

    def _fuse_sliding_attention(self, x0_pred, current_log_snr):
        history = list(self._history_buffer)
        K = len(history)
        if K == 1:
            return history[0][0]
        rms_dims = tuple(range(1, x0_pred.dim()))
        aligned_tensors = []
        for i in range(K):
            h_cur, snr_cur = history[i]
            if i < K - 1:
                h_older, snr_older = history[i + 1]
                gamma = 0.5 * (1.0 + math.exp(snr_older - snr_cur))
                rms_cur = torch.sqrt(torch.mean(h_cur ** 2, dim=rms_dims, keepdim=True) + 1e-08)
                rms_old = torch.sqrt(torch.mean(h_older ** 2, dim=rms_dims, keepdim=True) + 1e-08)
                h_older_aligned = h_older * (rms_cur / rms_old)
                h_aligned = h_cur + (h_cur - h_older_aligned) * gamma
            else:
                h_aligned = h_cur
            aligned_tensors.append(h_aligned)
        smooth_x_sq = self._smooth(x0_pred ** 2, kernel_size=3).sum(dim=1) + 1e-08
        sims = []
        for h in aligned_tensors:
            cos_sim = self._smooth(x0_pred * h, kernel_size=3).sum(dim=1) / (torch.sqrt(smooth_x_sq * self._smooth(h ** 2, kernel_size=3).sum(dim=1) + 1e-08) + 1e-08)
            sims.append(cos_sim)
        sims_t = torch.stack(sims, dim=1)
        z_scores = (sims_t - sims_t.mean(dim=1, keepdim=True)) / (sims_t.std(dim=1, keepdim=True) + 1e-06)
        delta_snrs = torch.tensor([abs(current_log_snr - h[1]) for h in history], dtype=x0_pred.dtype, device=x0_pred.device)
        delta_snrs = delta_snrs.view([1, K] + [1] * (x0_pred.dim() - 2))
        penalty = (1.0 + math.exp(current_log_snr)) * delta_snrs
        logits = 4.0 * z_scores - penalty
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        x0_cons = (torch.stack(aligned_tensors, dim=1) * torch.softmax(logits, dim=1).unsqueeze(2)).sum(dim=1)
        return x0_cons

    def apply(self, x0_pred, t_scalar):
        orig = x0_pred
        x0_pred = self._to_spatial(x0_pred)
        current_log_snr = self._log_snr(t_scalar)
        if self._history_buffer is None:
            self._history_buffer = collections.deque(maxlen=self.window_size)
            self._history_buffer.append((x0_pred.detach().clone(), current_log_snr))
            return self._from_spatial(x0_pred, orig)
        x0_cons = self._fuse_sliding_attention(x0_pred, current_log_snr)
        w_scalar = self.guide_scale - 1.0
        if w_scalar > 0.0001:
            raw_guidance = x0_pred - x0_cons
            proj_dims = tuple(range(1, x0_pred.dim()))
            dot_product = (raw_guidance * x0_pred).sum(dim=proj_dims, keepdim=True)
            norm_sq = (x0_pred ** 2).sum(dim=proj_dims, keepdim=True) + 1e-08
            g_parallel = dot_product / norm_sq * x0_pred
            g_ortho = raw_guidance - g_parallel
            g_low = self._smooth(g_ortho, kernel_size=5)
            g_high = g_ortho - g_low
            dynamic_high_freq_weight = 1.0 / (1.0 + math.exp(-current_log_snr))
            dampened_ortho_guidance = g_low + dynamic_high_freq_weight * g_high
            x0_out = x0_pred + w_scalar * dampened_ortho_guidance
        else:
            x0_out = x0_pred
        self._history_buffer.append((x0_pred.detach().clone(), current_log_snr))
        return self._from_spatial(x0_out, orig)
def candidates_and_weights(current, history, ell, rule):
    """Preserve the REAL old-to-new candidate construction unless stated.

The original array runs oldest first. Its difference extends OLDER away from
NEWER; naming it a proven current-time transport would be incorrect.
"""
    candidates = []
    dims = tuple(range(1, current.ndim))
    for i, (old, s) in enumerate(history):
        if i + 1 < len(history):
            newer, snew = history[i + 1]
            gamma = 0.5 * (1.0 + math.exp(snew - s))
            rms = (old.square().mean(dims, keepdim=True) + 1e-08).sqrt()
            rmsnew = (newer.square().mean(dims, keepdim=True) + 1e-08).sqrt()
            old = old + (old - newer * (rms / rmsnew)) * gamma
        candidates.append(old)
    pool = lambda x: F.avg_pool2d(x, 3, 1, 1)
    xnorm = pool(current.square()).sum(1) + 1e-08
    similarities = []
    for h in candidates:
        similarities.append(pool(current * h).sum(1) / ((xnorm * pool(h.square()).sum(1) + 1e-08).sqrt() + 1e-08))
    scores = torch.stack(similarities, 1)
    zs = (scores - scores.mean(1, keepdim=True)) / (scores.std(1, keepdim=True) + 1e-06)
    distances = torch.tensor([abs(ell - s) for _, s in history], device=current.device, dtype=current.dtype)
    distances = distances.view(1, -1, 1, 1)
    logits = 4 * (scores if rule == 'absolute_compat' else zs) - (1 + math.exp(ell)) * distances
    if rule == 'time_balanced':
        counts = [sum((abs(s - other) < 1e-06 for _, other in history)) for _, s in history]
        logits = logits - torch.tensor(counts, device=current.device, dtype=current.dtype).log().view(1, -1, 1, 1)
    logits = logits - logits.max(1, keepdim=True).values
    return (torch.stack(candidates, 1), logits.softmax(1), scores)
