"""Repeated-time observation tracking used by the fixed SiT Heun adapter."""
import torch

class SameTimePairs:
    """Adjacent equal-time RAW predictions supply a state-response direction.

    The latest completed pair is reused until the next pair is available.
    On fixed Heun2 the equal-time calls are a provisional endpoint followed by
    the next accepted-start evaluation. No solver state is changed. This is a
    view of state sensitivity, not two independent estimates of a static truth.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.response = None
        self.response_time = None
        self.pair_count = 0
        self.fresh = False

    def observe(self, raw, time):
        self.fresh = False
        if self.previous is not None:
            old, old_time = self.previous
            if old.shape != raw.shape:
                raise ValueError('Batch shape changed without resetting pair history')
            if float(time) == old_time:
                self.response = (raw - old).detach().clone()
                self.response_time = float(time)
                self.pair_count += 1
                self.fresh = True
        self.previous = (raw.detach().clone(), float(time))
        return self.response
