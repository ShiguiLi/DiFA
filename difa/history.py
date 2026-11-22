# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
"""Explicit storage of raw data predictions for temporal reference construction."""
from collections import deque
from typing import NamedTuple, Optional

import torch


class PredictionRecord(NamedTuple):
    prediction: torch.Tensor
    log_snr: float
    time: Optional[float] = None


class PredictionHistory:
    """Bounded raw history whose commit points are controlled by the adapter."""

    def __init__(self, window_size=3):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.records = deque(maxlen=window_size)
        self.commits = 0

    def record(self, prediction, log_snr, time=None):
        """Copy a raw prediction without adding it to the committed history."""
        return PredictionRecord(
            prediction.detach().clone(), float(log_snr),
            None if time is None else float(time),
        )

    def snapshot(self):
        """Return a tuple of records; treat the stored tensors as read-only."""
        return tuple(self.records)

    def commit(self, *records):
        self.records.extend(records)
        self.commits += len(records)

    def reset(self):
        self.records.clear()
        self.commits = 0
