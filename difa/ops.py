"""Model-independent grouped-tensor primitives, retaining measured arithmetic."""
import math
import torch

def branch_unit(q):
    combined = q.sum(1)
    length = combined.norm(dim=-1, keepdim=True)
    return combined / length.clamp_min(torch.finfo(q.dtype).tiny)

def branch_map(values, u, angular):
    """Exact inspected B03 prox; angular=True restores each input patch norm.

    values [B,K,T,D], u [B,T,D]. Positive homogeneity retains the scale of
    contributions; suppressed histories are never individually amplified.
    """
    direction = u[:, None]
    coefficient = (values * direction).sum(-1, keepdim=True).clamp_min(0)
    prox = values - coefficient / (1 + direction.square().sum(-1, keepdim=True)) * direction
    if not angular:
        return prox
    before = values.norm(dim=-1, keepdim=True)
    after = prox.norm(dim=-1, keepdim=True)
    return prox * (before / after.clamp_min(torch.finfo(values.dtype).tiny))

def bound_amplitude(candidate, reference, mode):
    if mode not in {'match', 'cap'}:
        raise ValueError(mode)
    tiny = torch.finfo(candidate.dtype).tiny
    cn = candidate.norm(dim=-1, keepdim=True)
    rn = reference.norm(dim=-1, keepdim=True)
    if mode == 'match':
        viable = cn > math.sqrt(torch.finfo(candidate.dtype).eps) * rn
        output = torch.where(viable, candidate * (rn / cn.clamp_min(tiny)), reference)
        fallback = float((~viable & (rn > 0)).float().mean())
    else:
        output = candidate * torch.minimum(torch.ones_like(cn), rn / cn.clamp_min(tiny))
        fallback = 0.0
    output = torch.where(rn > 0, output, torch.zeros_like(output))
    on = output.norm(dim=-1, keepdim=True)
    relative = (on - rn) / rn.clamp_min(tiny)
    energy = reference.square().sum()
    cosine = (output * reference).sum(-1) / (on.squeeze(-1) * rn.squeeze(-1)).clamp_min(tiny)
    valid = (on.squeeze(-1) > 0) & (rn.squeeze(-1) > 0)
    info = dict(amplitude_fallback_fraction=fallback, natural_below_L01_fraction=float(((cn < rn) & (rn > 0)).float().mean()), natural_above_L01_fraction=float(((cn > rn) & (rn > 0)).float().mean()), final_norm_overshoot_max=float(relative.clamp_min(0).max()), final_norm_relative_error_max=float(relative.abs().max()), final_cosine_mean=float(cosine[valid].clamp(-1, 1).mean()) if bool(valid.any()) else None, natural_RMS_over_L01=float((candidate.square().sum() / energy).sqrt()) if float(energy) > 0 else None, final_RMS_over_L01=float((output.square().sum() / energy).sqrt()) if float(energy) > 0 else None)
    return (output, info)

def reduce_chunks(rows):
    """Means are sample-weighted; *_max is a true maximum across chunks."""
    result = {}
    keys = set().union(*(row.keys() for _, row in rows))
    for key in keys:
        values = [(n, row[key]) for n, row in rows if row.get(key) is not None]
        if not values:
            result[key] = None
            continue
        if isinstance(values[0][1], list):
            result[key] = [sum((n * v[i] for n, v in values)) / sum((n for n, _ in values)) for i in range(len(values[0][1]))]
        elif key.endswith('_max'):
            result[key] = max((v for _, v in values))
        else:
            result[key] = sum((n * v for n, v in values)) / sum((n for n, _ in values))
    return result
def joint_reference(values, total, response, u):
    """Unique convex minimizer for EACH weighted guidance contribution.

min_h .5||h-g_i||^2 + .5(h.u)_+^2 + .5(h.m)^2/a,
a=||sum_i g_i||^2+||m||^2 (fixed at original query).

The response-only solution determines the branch active set; an active
branch uses the exact rank-two Woodbury solve. u need not have unit length.
No alternating projections or iteration threshold. No norm bound is claimed
for the sum followed by the inherited norm match.
"""
    tiny = torch.finfo(values.dtype).tiny
    a = total.square().sum(-1, keepdim=True) + response.square().sum(-1, keepdim=True)
    v = torch.where(a > 0, response / a.sqrt().clamp_min(tiny), torch.zeros_like(response))
    u2 = u.square().sum(-1, keepdim=True)
    v2 = v.square().sum(-1, keepdim=True)
    uv = (u * v).sum(-1, keepdim=True)
    gu = (values * u[:, None]).sum(-1, keepdim=True)
    gv = (values * v[:, None]).sum(-1, keepdim=True)
    inactive = values - gv / (1 + v2[:, None]) * v[:, None]
    active = (inactive * u[:, None]).sum(-1, keepdim=True) > 0
    determinant = (1 + u2) * (1 + v2) - uv.square()
    coeff_u = ((1 + v2[:, None]) * gu - uv[:, None] * gv) / determinant[:, None]
    coeff_v = ((1 + u2[:, None]) * gv - uv[:, None] * gu) / determinant[:, None]
    both = values - coeff_u * u[:, None] - coeff_v * v[:, None]
    result = torch.where(active, both, inactive)
    base = branch_map(values, u, False)
    result = torch.where((v2 == 0)[:, None], base, result)
    return (result, dict(joint_active_component_fraction=float(active.float().mean()), joint_determinant_min=float(determinant.min()), response_penalty_trace_mean=float(v2.mean())))
def comparison(candidate, baseline):
    energy = baseline.square().sum()
    bn = baseline.norm(dim=-1)
    cn = candidate.norm(dim=-1)
    valid = (bn > 0) & (cn > 0)
    cosine = (candidate * baseline).sum(-1) / (bn * cn).clamp_min(torch.finfo(candidate.dtype).tiny)
    relative = (cn - bn).abs() / bn.clamp_min(torch.finfo(candidate.dtype).tiny)
    return dict(final_change_over_baseline_RMS=float(((candidate - baseline).square().sum() / energy).sqrt()) if float(energy) > 0 else None, final_RMS_over_baseline=float((candidate.square().sum() / energy).sqrt()) if float(energy) > 0 else None, final_direction_cosine_mean=float(cosine[valid].clamp(-1, 1).mean()) if bool(valid.any()) else None, final_group_norm_relative_error_max=float(relative[bn > 0].max()) if bool((bn > 0).any()) else 0.0)
