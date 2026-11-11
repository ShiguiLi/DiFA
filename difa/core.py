"""DiFA prediction refinement on generic grouped tensors.

No model class, image dimensions, time schedule, CFG convention or solver
appears in this module. Inputs must already share the guidance coordinate.
The reference builder and model adapter own how those tensors are obtained.
"""
from dataclasses import dataclass
import torch
from .ops import branch_unit,branch_map,bound_amplitude,joint_reference,comparison


@dataclass
class Refinement:
    guidance: torch.Tensor
    diagnostics: dict


class DiFA:
    """Stateless G03 refinement, reusable across models and tensor layouts.

    base_guidance: [B,G,D], actual reference-guidance tensor before refinement.
    contributions: [B,K,G,D] or None when no reference history is available.
    branch_responses: [B,R,G,D], R>=1 in the same coordinate as guidance.
    state_response: [B,G,D] or None when no valid response has been observed.

    G indexes arbitrary feature groups, not necessarily image patches. K and R
    are not fixed to SiT. This class performs no model calls, history commits,
    RNG draws or optimization of trainable parameters. The SiT adapter uses
    K<=3 and R=2. Other signal constructions require their own validation.
    """

    @staticmethod
    def baseline(base_guidance, branch_responses):
        """Norm-restored joint branch prox for any positive number of branches."""
        g,q=base_guidance,branch_responses
        if g.ndim!=3 or q.ndim!=4 or q.shape[1]<1 or q.shape[0]!=g.shape[0] or q.shape[2:]!=g.shape[1:]:
            raise ValueError('Expected guidance[B,G,D] and branches[B,R,G,D], R>=1')
        if g.dtype!=q.dtype or g.device!=q.device or not g.is_floating_point():
            raise ValueError('Guidance and responses must share floating dtype/device')
        tiny=torch.finfo(g.dtype).tiny
        total=q.norm(dim=-1).sum(dim=1)
        combined=q.sum(dim=1)
        length=combined.norm(dim=-1,keepdim=True)
        vector=combined/length.clamp_min(tiny)
        vector=torch.where(length>0,vector,torch.zeros_like(vector))
        coefficient=(g*vector).sum(-1,keepdim=True).clamp_min(0)
        coefficient=coefficient/(1+vector.square().sum(-1,keepdim=True))
        prox=torch.where((total>0)[...,None],g-coefficient*vector,g)
        before=g.norm(dim=-1,keepdim=True);after=prox.norm(dim=-1,keepdim=True)
        if bool(((before>0)&(after==0)).any()):
            raise RuntimeError('Nonzero guidance lost in joint branch prox')
        ratio=torch.where(before>0,before/after.clamp_min(tiny),torch.ones_like(before))
        changed=prox*ratio
        return torch.where((total>0)[...,None],changed,g)

    @torch.no_grad()
    def refine(self,base_guidance,contributions,branch_responses,state_response=None,*,baseline=None):
        """Return a correction tensor and diagnostics without changing inputs.

        Keep separate incremental additions: floating-point preservation of
        the measured parent path matters even when they simplify algebraically.
        `baseline` may cache self.baseline(...) from the same inputs only.
        """
        g,q=base_guidance,branch_responses
        if baseline is None:baseline=self.baseline(g,q)
        if contributions is None:return Refinement(baseline,{})
        if contributions.ndim!=4 or contributions.shape[0]!=g.shape[0] or contributions.shape[1]<1 or contributions.shape[2:]!=g.shape[1:]:
            raise ValueError('contributions must have shape [B,K,G,D], K>=1')
        if contributions.dtype!=g.dtype or contributions.device!=g.device:
            raise ValueError('Contributions must share guidance dtype/device')
        if state_response is not None and (state_response.shape!=g.shape or state_response.dtype!=g.dtype or state_response.device!=g.device):
            raise ValueError('state_response must match guidance shape/dtype/device')
        u=branch_unit(q)
        total=contributions.sum(1)
        analytic_reference=branch_map(total[:,None],u,True)[:,0]
        distributed=branch_map(contributions,u,False).sum(1)
        proposal=baseline+(distributed-analytic_reference)
        parent,_=bound_amplitude(proposal,baseline,'match')
        if state_response is None:
            value=parent;info=dict(pair_available=False)
        else:
            altered,info=joint_reference(contributions,total,state_response,u)
            joint_sum=altered.sum(1)
            delta=joint_sum-distributed
            value,amp=bound_amplitude(proposal+delta,baseline,'match')
            energy=total.square().sum();base_energy=distributed.square().sum()
            info.update(pair_available=True,mapped_response_RMS=float(state_response.square().mean().sqrt()),
                mapped_response_over_contrast_RMS=float((state_response.square().sum()/energy).sqrt()) if float(energy)>0 else None,
                pre_norm_change_over_M04_RMS=float((delta.square().sum()/base_energy).sqrt()) if float(base_energy)>0 else None)
            info.update(amp)
        info.update(comparison(value,parent))
        if not bool(torch.isfinite(value).all()):raise RuntimeError('Nonfinite DiFA guidance')
        return Refinement(value,info)

    __call__=refine
