"""SiT-specific terminal branch responses; used only by the SiT adapter."""
import math
import torch
from torch import nn
import torch.nn.functional as F

def head_direction(head, final_hidden, residual_update, condition):
    """Exact J_head(h_L,c) @ delta, including the LayerNorm derivative.

    This is local readout sensitivity of a recorded hidden residual, NOT the
    total derivative through subsequent transformer blocks and NOT an error
    estimate. It remains useful for ranking candidate output directions.
    """
    if head.norm_final.elementwise_affine:
        raise ValueError('Expected the inspected SiT non-affine final LayerNorm')
    if residual_update.shape != final_hidden.shape:
        raise ValueError('Residual and final hidden shapes must match')
    centered = final_hidden - final_hidden.mean(-1, keepdim=True)
    variance_eps = centered.square().mean(-1, keepdim=True) + head.norm_final.eps
    delta_centered = residual_update - residual_update.mean(-1, keepdim=True)
    normalized_delta = (delta_centered - centered * (centered * residual_update).mean(-1, keepdim=True) / variance_eps) / variance_eps.sqrt()
    _, scale = head.adaLN_modulation(condition).chunk(2, dim=1)
    return F.linear(normalized_delta * (1 + scale[:, None]), head.linear.weight, None)

def patch_readout_to_image(tokens, patch_size, all_output_channels, kept_channels):
    """Match SiT's p,p,C head ordering, then discard unused output channels."""
    b, length, width = tokens.shape
    side = int(length ** 0.5)
    if side * side != length or width != patch_size ** 2 * all_output_channels:
        raise ValueError('Unexpected head token geometry')
    if not 0 < kept_channels <= all_output_channels:
        raise ValueError('Invalid channel selection')
    grid = tokens.reshape(b, side, side, patch_size, patch_size, all_output_channels)
    return grid[..., :kept_channels].permute(0, 5, 1, 3, 2, 4).reshape(b, kept_channels, side * patch_size, side * patch_size)
def _identity_in_eval(module, name):
    if module is None or isinstance(module, nn.Identity):
        return
    if isinstance(module, nn.Dropout) and (not module.training):
        return
    raise ValueError(f'Unsupported non-identity terminal MLP {name}: {type(module).__name__}')

def _validate(block, head, before, middle, after, condition):
    if block.training or head.training or block.mlp.training:
        raise ValueError('Branch responses require the frozen model in eval mode')
    if before.shape != middle.shape or after.shape != middle.shape or middle.ndim != 3:
        raise ValueError('Require matching [batch, tokens, hidden] block states')
    if condition.ndim != 2 or condition.shape[0] != middle.shape[0]:
        raise ValueError('Condition batch does not match block states')
    norm = block.norm2
    if not isinstance(norm, nn.LayerNorm) or norm.elementwise_affine:
        raise ValueError('Expected the SiT non-affine terminal norm2')
    if tuple(norm.normalized_shape) != (middle.shape[-1],):
        raise ValueError('Expected LayerNorm on the hidden dimension only')
    mlp = block.mlp
    if not isinstance(mlp.fc1, nn.Linear) or not isinstance(mlp.fc2, nn.Linear):
        raise ValueError('Expected terminal MLP with two Linear layers')
    if not isinstance(mlp.act, nn.GELU) or mlp.act.approximate not in {'none', 'tanh'}:
        raise ValueError('Expected GELU with none/tanh approximation')
    _identity_in_eval(getattr(mlp, 'norm', None), 'norm')
    for name in ('drop', 'drop1', 'drop2'):
        _identity_in_eval(getattr(mlp, name, None), name)

def _layer_norm_direction(value, tangent, eps):
    centered = value - value.mean(-1, keepdim=True)
    variance_eps = centered.square().mean(-1, keepdim=True) + eps
    tangent_centered = tangent - tangent.mean(-1, keepdim=True)
    return (tangent_centered - centered * (centered * tangent).mean(-1, keepdim=True) / variance_eps) / variance_eps.sqrt()

def _gelu_derivative(value, approximate):
    if approximate == 'none':
        return 0.5 * (1 + torch.erf(value / math.sqrt(2))) + value * torch.exp(-0.5 * value.square()) / math.sqrt(2 * math.pi)
    squared = value.square()
    factor = math.sqrt(2 / math.pi)
    transformed = torch.tanh(factor * value * (1 + 0.044715 * squared))
    return 0.5 * (1 + transformed) + 0.5 * value * (1 - transformed.square()) * factor * (1 + 0.134145 * squared)

@torch.no_grad()
def coupled_head_directions(block, head, before, middle, after, condition):
    """Return exact first derivatives for terminal residual gate scaling.

    h_mid = h_before + attention_delta
    h_after = h_mid + M(h_mid)

    Attention: D head(h_after)[attention_delta + D M(h_mid)[attention_delta]]
    MLP:       D head(h_after)[h_after - h_mid]

    Outputs are final-head patch tensors, before unpatchification/CFG/coordinate
    conversion. Caller chunks the input batch (normally HEAD_CHUNK=125).
    """
    _validate(block, head, before, middle, after, condition)
    attention_delta = middle - before
    _, _, _, shift, scale, gate = block.adaLN_modulation(condition).chunk(6, dim=1)
    normalized = block.norm2(middle) * (1 + scale[:, None]) + shift[:, None]
    normalized_tangent = _layer_norm_direction(middle, attention_delta, block.norm2.eps) * (1 + scale[:, None])
    mlp = block.mlp
    preactivation = mlp.fc1(normalized)
    tangent = F.linear(normalized_tangent, mlp.fc1.weight, None)
    tangent = tangent * _gelu_derivative(preactivation, mlp.act.approximate)
    mlp_response = F.linear(tangent, mlp.fc2.weight, None) * gate[:, None]
    return {'attn': head_direction(head, after, attention_delta + mlp_response, condition), 'mlp': head_direction(head, after, after - middle, condition)}
