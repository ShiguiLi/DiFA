"""SiT-specific prediction coordinates, branch observations and Heun lifecycle.

The actual DiFA tensor algorithm lives in ..core. This adapter is the only
place that knows SiT CFG channels, null conditioning or the solver schedule.
"""
import math
import os
import torch
from ..core import DiFA
from ..reference import SpatialReference
from ..ops import reduce_chunks
from .observer import SiTObserverAdapter
from .pairs import SameTimePairs
from .sit_response import patch_readout_to_image,coupled_head_directions


def cfg_output(raw,scale):
    eps,rest=raw[:,:3],raw[:,3:]
    cond,uncond=torch.split(eps,len(eps)//2,dim=0)
    guided=uncond+scale*(cond-uncond)
    return torch.cat([torch.cat([guided,guided],0),rest],1)


class TerminalResponses:
    def __init__(self,model,chunk):
        self.model=model;self.chunk=chunk;self.directions={};self.mlp_jvp_chunks=0

    def wants(self,index,total):return index==total-1

    def block(self,index,before,middle,after,condition):
        parts={'attn':[],'mlp':[]}
        for lo in range(0,len(after),self.chunk):
            hi=lo+self.chunk
            result=coupled_head_directions(self.model.blocks[index],self.model.final_layer,
                before[lo:hi],middle[lo:hi],after[lo:hi],condition[lo:hi])
            for name,value in result.items():
                parts[name].append(patch_readout_to_image(value,self.model.patch_size,
                    self.model.out_channels,self.model.in_channels))
            self.mlp_jvp_chunks+=1
        self.directions={name:torch.cat(values,0).detach() for name,values in parts.items()}

    def finish(self,features,condition,raw):pass


class SiTG03Guidance:
    """SiT callable using the generic DiFA core and spatial reference builder.

    Signature: prediction=guidance(x,t,y,cfg_scale=1.5).
    Defaults reproduce measured G03 on SiT-XL/2 using fixed forward Heun2.
    Instantiate per trajectory/batch or reset() before reuse. Model stays eval
    and unmodified. Other models should integrate DiFA directly with their own
    prediction-coordinate and response adapters, not inherit SiT assumptions.
    """
    def __init__(self,model,guide_scale=1.75,window_size=3,chunk_size=None):
        if model.training:raise ValueError('Call model.eval() before creating guidance')
        self.model=model;self.observed_model=SiTObserverAdapter(model)
        self.scale=guide_scale
        self.chunk=max(1,int(os.environ.get('HEAD_CHUNK','125'))) if chunk_size is None else max(1,int(chunk_size))
        self.algorithm=DiFA()
        self.reference=SpatialReference(guide_scale,window_size,model.patch_size)
        self.reset()

    def reset(self):
        self.reference.reset();self.pairs=SameTimePairs()
        self.pending=self.last_end=self.condition=None
        self.stats=dict(calls=0,intervals=0,raw_commits=0,trial_commits=0,mlp_jvp_chunks=0,trace=[])

    def _validate(self,t,y):
        scalar=float(t[0])
        if not math.isfinite(scalar) or not bool((t==t[0]).all()):
            raise ValueError('SiT adapter requires synchronized fixed Heun times')
        if self.condition is not None and not torch.equal(self.condition,y):
            raise ValueError('Condition changed without reset')
        if self.pending is None:
            if self.last_end is None:
                if scalar!=0.:raise ValueError('This SiT adapter starts at t=0')
            elif abs(scalar-self.last_end)>2e-7:
                raise ValueError('Expected start evaluation at the previous endpoint time')
        elif scalar<=self.pending['t']:
            raise ValueError('Expected provisional endpoint at a later time')
        return scalar

    @torch.no_grad()
    def __call__(self,x,t,y,cfg_scale=1.5):
        scalar=self._validate(t,y);start_stage=self.pending is None
        n=len(x)//2
        observer=TerminalResponses(self.model,self.chunk)
        raw=self.observed_model(torch.cat([x[:n],x[:n]],0),t,y,structure_observer=observer)
        prediction=cfg_output(raw,cfg_scale);tv=max(scalar,1e-6)
        first,second=(x-tv*prediction).chunk(2,0)
        response=self.pairs.observe(first,scalar)
        record=self.reference.record(first,scalar)
        snapshot=self.reference.snapshot() if start_stage else self.pending['snapshot']
        original=self.reference.evaluate(first,scalar,snapshot)
        g0=self.reference.to_groups(original-first)
        q=torch.stack([self.reference.to_groups(-tv*cfg_output(observer.directions[name],cfg_scale)[:n])
                       for name in ('attn','mlp')],1)
        baseline=self.algorithm.baseline(g0,q)
        info=dict(pair_available=response is not None)
        if not snapshot or self.scale<=1.:changed=baseline
        else:
            pieces=[];rows=[]
            for lo in range(0,n,self.chunk):
                hi=min(n,lo+self.chunk)
                history=[(v[lo:hi],s,time) for v,s,time in snapshot]
                components=self.reference.contributions(first[lo:hi],scalar,history)
                mapped_response=self.reference.map_response(first[lo:hi],scalar,None if response is None else response[lo:hi])
                result=self.algorithm.refine(g0[lo:hi],components,q[lo:hi],mapped_response,baseline=baseline[lo:hi])
                pieces.append(result.guidance);rows.append((hi-lo,result.diagnostics))
            changed=torch.cat(pieces,0);info=reduce_chunks(rows)
        out=self.reference.add_guidance(first,changed)
        result=(x-torch.cat([out,second],0))/tv
        if self.scale<=1.:result=prediction
        if not bool(torch.isfinite(result).all()):raise RuntimeError('Nonfinite DiFA prediction; no raw commit')
        self.stats['calls']+=1;self.stats['mlp_jvp_chunks']+=observer.mlp_jvp_chunks
        if start_stage:
            self.pending=dict(t=scalar,record=record,snapshot=snapshot)
            if self.condition is None:self.condition=y.detach().clone()
        else:
            self.reference.commit(self.pending['record'],record)
            self.stats['trial_commits']+=1;self.stats['intervals']+=1
            self.stats['raw_commits']=self.reference.commits;self.last_end=scalar;self.pending=None
        self.stats['trace'].append(dict(t=scalar,stage='accepted_start' if start_stage else 'provisional_endpoint',
            read_history_times=[r[2] for r in snapshot],read_history_count=len(snapshot),
            pair_count=self.pairs.pair_count,pair_fresh=self.pairs.fresh,pair_time=self.pairs.response_time,
            pair_age=scalar-self.pairs.response_time if self.pairs.response_time is not None else None,
            mlp_jvp_chunks=observer.mlp_jvp_chunks,**info))
        return result


G03Guidance=SiTG03Guidance
