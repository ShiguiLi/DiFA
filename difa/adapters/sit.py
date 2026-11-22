# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShiguiLi
"""SiT prediction conversion and Euler/Heun lifecycle for DiFA."""
import math
import torch

from difa import DiFA, PredictionHistory, PredictionRecord, LogSNRMemory
from difa.memory import standardize_components
from difa._impl.components import component_statistics


def _rms(value):
    return float(value.square().mean().sqrt())


def _cfg_output(raw, scale):
    eps, rest = raw[:, :3], raw[:, 3:]
    conditional, unconditional = torch.split(eps, len(eps) // 2, dim=0)
    guided = unconditional + scale * (conditional - unconditional)
    return torch.cat([torch.cat([guided, guided], 0), rest], 1)

def _log_snr(time):
    value = max(1e-6, min(1. - 1e-6, time))
    return 2. * (math.log(value) - math.log1p(-value))

def _merge_stats(rows):
    result = {}
    for key in set().union(*(stats.keys() for _, stats in rows)):
        values = [(n, stats[key]) for n, stats in rows if stats.get(key) is not None]
        if not values:
            result[key] = None
        elif key.endswith("_min"):
            result[key] = min(value for _, value in values)
        elif key.endswith("_max"):
            result[key] = max(value for _, value in values)
        else:
            result[key] = sum(n * value for n, value in values) / sum(n for n, _ in values)
    return result

class SiTDiFA:
    """Wrap an eval-mode SiT model with the selected DiFA refinement.

    Use a new instance per independent batch, or call reset(). The solver
    must match this adapter's fixed forward Euler/Heun evaluation order.
    """

    def __init__(self, model, guide_scale=1.75, window_size=3, chunk_size=None, *, solver='heun'):
        if model.training:
            raise ValueError('Call model.eval() before creating SiTDiFA')
        if solver not in {'euler', 'heun'}:
            raise ValueError('solver must be euler or heun')
        self.model = model
        self.solver = solver
        self.scale = guide_scale
        self.chunk = 125 if chunk_size is None else max(1, int(chunk_size))
        self.algorithm = DiFA(guide_scale, model.patch_size)
        self.method = 'difa'
        self.reference = PredictionHistory(window_size)
        self.reset()

    def reset(self):
        self.reference.reset()
        self.pending = self.last_end = self.condition = None
        self.stats = dict(calls=0, intervals=0, raw_commits=0, trace=[])
        self.memory_state = LogSNRMemory()
        self.peer_memory_states = (LogSNRMemory(), LogSNRMemory())
        self.response_memory_state = LogSNRMemory()
        self.memory_last_time = self.cfg_value = None


    def _validate(self, t, y):
        scalar = float(t[0])
        if not math.isfinite(scalar) or not 0. <= scalar <= 1. or not bool((t == t[0]).all()):
            raise ValueError("Expected synchronized solver times in [0, 1]")
        if self.condition is not None and not torch.equal(self.condition, y):
            raise ValueError("Condition changed without reset")
        if self.solver == "euler":
            if self.last_end is None and scalar != 0.:
                raise ValueError("This SiT adapter starts at t=0")
            if self.last_end is not None and scalar <= self.last_end:
                raise ValueError("Expected increasing Euler evaluation times")
            return scalar
        if self.pending is None:
            if self.last_end is None:
                if scalar != 0.:
                    raise ValueError("This SiT adapter starts at t=0")
            elif abs(scalar - self.last_end) > 2e-7:
                raise ValueError("Expected start evaluation at the previous endpoint time")
        elif scalar <= self.pending["t"]:
            raise ValueError("Expected provisional endpoint at a later time")
        return scalar

    def _standardized(self, data):
        pieces = [standardize_components(data[lo:lo + self.chunk])
                  for lo in range(0, len(data), self.chunk)]
        return tuple(torch.cat([item[index] for item in pieces], 0)
                     for index in (0, 1))

    def _commit_views(self, records, shapes, peer_shapes, response_parts):
        states = (self.memory_state, *self.peer_memory_states, self.response_memory_state)
        before = tuple(state.snapshot() for state in states)
        try:
            for index, record in enumerate(records):
                self.memory_state.commit(record.log_snr, shapes[index])
                for state, observed in zip(self.peer_memory_states, peer_shapes[index]):
                    state.commit(record.log_snr, observed)
                self.response_memory_state.commit(record.log_snr,response_parts[index])
            self.reference.commit(*records)
        except Exception:
            for state, snapshot in zip(states, before):
                state.restore(snapshot)
            raise
        self.memory_last_time = records[-1].time
        self.stats['raw_commits'] = self.reference.commits

    @torch.no_grad()
    def __call__(self, x, t, y, cfg_scale=1.5):
        scalar = self._validate(t, y)
        if not math.isfinite(float(cfg_scale)):
            raise ValueError('CFG scale must be finite')
        if self.cfg_value is not None and float(cfg_scale) != self.cfg_value:
            raise ValueError('CFG scale changed without reset')
        start = self.pending is None
        n = len(x)//2
        remaining = 1.-scalar
        raw = self.model(torch.cat([x[:n], x[:n]], 0), t, y)
        prediction = _cfg_output(raw, cfg_scale)
        data = x[:n]+remaining*prediction[:n]
        # Preserve the existing CFG arithmetic and its first-three-channel mask.
        # The fourth-channel base remains conditional, not raw null output.
        contrast = torch.zeros_like(data)
        contrast[:, :3] = remaining*(raw[:n, :3]-raw[n:, :3])
        peers = (data-(cfg_scale-1.)*contrast, data-cfg_scale*contrast)
        mixing = data.new_ones((1, data.shape[1], 1, 1))
        mixing[:, :3] = cfg_scale
        ell = _log_snr(scalar)
        record = self.reference.record(data, ell, scalar)
        shapes = self._standardized(data)
        peer_shapes = tuple(self._standardized(value) for value in peers)
        # Raw low/residual response observations, without moment normalization.
        response_parts = tuple(part for part,_,_ in component_statistics(peers[0]-peers[1]))
        snapshot = self.reference.snapshot() if start else self.pending['snapshot']
        memory_snapshot = self.memory_state.snapshot() if start else self.pending['memory_snapshot']
        peer_snapshots = tuple(state.snapshot() for state in self.peer_memory_states) if start else self.pending['peer_snapshots']
        response_snapshot = self.response_memory_state.snapshot() if start else self.pending['response_snapshot']
        memory, mass = memory_snapshot[:2]
        response_memory,response_mass=response_snapshot[:2]
        if response_mass != mass:
            raise RuntimeError('Raw response-memory mass differs from joint memory')
        peer_memory = tuple(state[0] for state in peer_snapshots)
        peer_mass = peer_snapshots[0][1]
        if peer_snapshots[1][1] != peer_mass:
            raise RuntimeError('Branch raw-memory masses disagree')
        endpoint = scalar == 1.
        gate = 0. if endpoint else 1.-1./(1.+math.exp(-ell))
        info = {}
        if endpoint or self.scale <= 1.:
            result = prediction
        else:
            pieces, rows = [], []
            for lo in range(0, n, self.chunk):
                hi = min(n, lo+self.chunk)
                history = [PredictionRecord(value[lo:hi], level, time)
                           for value, level, time in snapshot]
                chunk_memory = tuple(value[lo:hi] for value in memory) if memory is not None else None
                chunk_peer_memory = tuple(tuple(value[lo:hi] for value in moments)
                                          for moments in peer_memory) if all(moments is not None for moments in peer_memory) else None
                chunk_response_memory=tuple(value[lo:hi] for value in response_memory) if response_memory is not None else None
                refined = self.algorithm.refine(data[lo:hi], history, log_snr=ell,
                        high_frequency_weight=gate, memory=chunk_memory, mass=mass,
                        peers=tuple(value[lo:hi] for value in peers),
                        peer_memory=chunk_peer_memory, peer_mass=peer_mass, mixing=mixing, response_memory=chunk_response_memory,response_mass=response_mass)
                pieces.append(refined.correction)
                rows.append((hi-lo, refined.diagnostics))
            increment = torch.cat(pieces, 0)
            info = _merge_stats(rows)
            result = torch.cat([prediction[:n]+increment, prediction[n:]], 0)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError('Nonfinite DiFA output; raw observations not committed')
        delta = result[:n]-prediction[:n]
        trace = dict(t=scalar, stage='euler' if self.solver == 'euler' else (
                        'accepted_start' if start else 'provisional_endpoint'),
                     method=self.method, data_log_snr=ell, data_gain=remaining,
                     high_frequency_gate=gate, endpoint_bypass=endpoint,
                     read_history_times=[entry[2] for entry in snapshot], read_history_count=len(snapshot),
                     applied_velocity_correction_RMS=_rms(delta),
                     applied_data_correction_RMS=_rms(remaining*delta),
                     memory_mass_read=mass, memory_intervals_read=memory_snapshot[4],
                     memory_raw_observations_read=memory_snapshot[5],
                     memory_last_time_read=self.memory_last_time,
                     views_peer_memory_mass_read=peer_mass,
                     views_peer_memory_intervals_read=peer_snapshots[0][4],
                     views_peer_raw_observations_read=peer_snapshots[0][5],
                     views_CFG_mixed_channels=3,
                     difa_memory_mass_read=response_mass,
                     difa_memory_intervals_read=response_snapshot[4],
                     difa_memory_observations_read=response_snapshot[5])
        trace.update(info)
        self.stats['calls'] += 1
        if self.condition is None:
            self.condition = y.detach().clone()
        self.cfg_value = float(cfg_scale)
        if self.solver == 'euler':
            self._commit_views((record,), (shapes,), (peer_shapes,), (response_parts,))
            self.stats['intervals'] += 1
            self.last_end = scalar
        elif start:
            self.pending = dict(t=scalar, record=record, shapes=shapes, peer_shapes=peer_shapes,
                                snapshot=snapshot, memory_snapshot=memory_snapshot,
                                peer_snapshots=peer_snapshots,response_snapshot=response_snapshot,response_parts=response_parts,
                                velocity=prediction[:n].detach().clone(), delta=delta.detach())
        else:
            step = scalar-self.pending['t']
            dv = (step/2.)*(prediction[:n]-self.pending['velocity'])
            db = (step/2.)*(delta-self.pending['delta'])
            trace.update(interval_dt=step, heun_euler_raw_RMS=_rms(dv),
                         heun_euler_guidance_RMS=_rms(db), heun_euler_total_RMS=_rms(dv+db))
            self._commit_views((self.pending['record'], record),
                               (self.pending['shapes'], shapes),
                               (self.pending['peer_shapes'], peer_shapes),
                               (self.pending['response_parts'],response_parts))
            self.stats['intervals'] += 1
            self.last_end = scalar
            self.pending = None
        trace.update(memory_mass_after=self.memory_state.mass,
                     memory_intervals_after=self.memory_state.intervals,
                     views_peer_memory_mass_after=self.peer_memory_states[0].mass,
                     difa_memory_mass_after=self.response_memory_state.mass)
        self.stats['trace'].append(trace)
        return result
