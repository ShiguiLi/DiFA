"""Model-independent spatial reference construction with explicit history.

This concrete builder handles [B,C,H,W] predictions. Other layouts can build
the grouped tensors consumed by DiFA directly without this spatial builder.
The caller decides which raw predictions to commit and when to freeze reads.
"""
from collections import deque
import torch
from .consensus import DiFA_FM,candidates_and_weights
from .geometry import image_to_patch_channels,patch_to_image,difa_linear,to_tokens


class SpatialReference:
    def __init__(self,guide_scale=1.75,window_size=3,group_size=2,log_snr=None):
        if group_size<1 or window_size<1:raise ValueError('Positive group/window size required')
        self.group_size=group_size
        self.scale=guide_scale
        self.consensus=DiFA_FM(guide_scale=guide_scale,window_size=window_size,rescale_phi=1.)
        # A path adapter may supply its own normalized-time -> log-SNR mapping.
        # Default retains the measured linear-path mapping exactly.
        if log_snr is not None:self.consensus._log_snr=log_snr
        self.records=deque(maxlen=window_size)
        self.commits=0

    def reset(self):
        self.records.clear();self.consensus.reset();self.commits=0

    def snapshot(self):
        return tuple(self.records)

    def record(self,prediction,time):
        if prediction.ndim!=4:raise ValueError('SpatialReference expects prediction[B,C,H,W]')
        return prediction.detach().clone(),self.consensus._log_snr(time),float(time)

    def commit(self,*records):
        """Commit only raw records made with record(); never refined predictions."""
        self.records.extend(records);self.commits+=len(records)

    def evaluate(self,prediction,time,history):
        """Original DiFA prediction, reading supplied history without committing."""
        saved=self.consensus._history_buffer
        self.consensus._history_buffer=(deque([(v,s) for v,s,_ in history],maxlen=self.records.maxlen) if history else None)
        try:return self.consensus.apply(prediction,time)
        finally:self.consensus._history_buffer=saved

    def to_groups(self,tensor):
        return image_to_patch_channels(tensor,self.group_size)

    def add_guidance(self,prediction,guidance):
        return prediction+patch_to_image(guidance,prediction.shape[2],prediction.shape[3],self.group_size)

    def contributions(self,prediction,time,history):
        """Decompose the raw-reference contrast into grouped DiFA contributions."""
        if not history:return None
        entries=[(v,s) for v,s,_ in history]
        ell=self.consensus._log_snr(time)
        if len(entries)==1:
            candidates=entries[0][0][:,None]
            weights=torch.ones_like(prediction[:,None,0])
        else:candidates,weights,_=candidates_and_weights(prediction,entries,ell,'legacy')
        residuals=prediction[:,None]-candidates
        return to_tokens(difa_linear(prediction,weights[:,:,None]*residuals,ell,self.scale),self.group_size)

    def map_response(self,prediction,time,response):
        if response is None:return None
        if response.shape!=prediction.shape:raise ValueError('Response must share prediction shape')
        return to_tokens(difa_linear(prediction,response[:,None],self.consensus._log_snr(time),self.scale),self.group_size)[:,0]
