"""CPU checks for the DiFA formulas and production-adapter lifecycle."""
import copy
import unittest
from unittest.mock import patch

import torch
from difa import DiFA, LogSNRMemory
from difa.memory import response_components, standardize_components
from difa._impl.response import response_decomposition
from difa.adapters.sit import SiTDiFA, _cfg_output
from sample import create_parser, resolve_sampling
from difa.sampling import build_prediction


class SmallNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.calls = 0

    def forward(self, x, t, y):
        self.calls += 1
        h, w = x.shape[-2:]
        px = torch.linspace(-1., 1., w, dtype=x.dtype, device=x.device)[None, None, None]
        py = torch.linspace(-1., 1., h, dtype=x.dtype, device=x.device)[None, None, :, None]
        channels = torch.arange(4, dtype=x.dtype, device=x.device)[None, :, None, None] * .27
        response = torch.tanh(.3 * x + torch.sin(px * 2.1 + py * 1.7 + channels))
        label = y[:, None, None, None].to(x) / 1000. - .5
        return (.23 * torch.tanh(self.conv(x)) + .1 * x
                + .08 * torch.sin(t[:, None, None, None] * 2.3) + .18 * label * response)

    def forward_with_cfg(self, x, t, y, cfg_scale=1.5):
        n = len(x) // 2
        return _cfg_output(self(torch.cat([x[:n], x[:n]]), t, y), cfg_scale)


def integrate(initial, prediction, labels, solver, nodes, observer=None):
    state = initial.clone()
    times = torch.linspace(0., 1., nodes)
    outputs = []
    for left, right in zip(times[:-1], times[1:]):
        start = state.clone()
        step = right - left
        t = left.expand(len(state))
        first = prediction(state, t, labels, 1.5)
        outputs.append(first.detach().clone())
        if observer is not None:
            observer(prediction, start, t, first)
        provisional = state + step * first
        if solver == 'euler':
            state = provisional
        else:
            t = right.expand(len(state))
            second = prediction(provisional, t, labels, 1.5)
            outputs.append(second.detach().clone())
            if observer is not None:
                observer(prediction, provisional, t, second)
            state = state + step * (first + second) / 2.
    return state, outputs


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


class DiFATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2)

    def inputs(self):
        data = torch.randn(3, 4, 8, 8)
        return torch.cat([data, data]), torch.tensor([0, 207, 999, 1000, 1000, 1000])

    def test_selected_core_and_independent_statistic_formula(self):
        self.assertEqual(DiFA().response_variant, 'difa')
        self.assertEqual(DiFA().parent.method, 'consensus')
        for dtype, tolerance in ((torch.float64, 1e-14), (torch.float32, 1e-6)):
            current = torch.randn(2, 4, 8, 8, dtype=dtype)
            past = torch.randn_like(current) * .7 + .2
            mass = .6
            moments = tuple(part * mass for part in response_components(past))
            full, shape, statistic, _ = response_decomposition(current, moments, mass)
            expected = torch.zeros_like(current)
            for now, moment in zip(response_components(current), moments):
                old = moment / mass
                mc = now.mean((-2, -1), keepdim=True)
                mh = old.mean((-2, -1), keepdim=True)
                sc = (now - mc).square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
                sh = (old - mh).square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
                expected += mc - mh + .5 * (sc - sh) * ((now - mc) / sc + (old - mh) / sh)
            torch.testing.assert_close(statistic, expected, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(full, shape + statistic, atol=tolerance, rtol=tolerance)

    def test_equal_statistics_remove_only_additional_statistic_signal(self):
        past = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        moments = response_components(past)
        _, _, stationary, _ = response_decomposition(past, moments, 1.)
        _, shape, statistic, _ = response_decomposition(past.flip(-1), moments, 1.)
        self.assertEqual(float(stationary.abs().max()), 0.)
        self.assertLess(float(statistic.abs().max()), 1e-14)
        self.assertGreater(float(shape.abs().max()), .1)

    def test_repeated_log_snr_does_not_add_mass(self):
        memory = LogSNRMemory()
        memory.commit(-2., standardize_components(torch.randn(2, 4, 8, 8)))
        memory.commit(-1., standardize_components(torch.randn(2, 4, 8, 8)))
        moments, mass, intervals = copy.deepcopy(memory.moments), memory.mass, memory.intervals
        memory.commit(-1., standardize_components(torch.randn(2, 4, 8, 8)))
        self.assertTrue(equal_state(moments, memory.moments))
        self.assertEqual((mass, intervals), (memory.mass, memory.intervals))

    def test_euler_and_heun_lifecycle_raw_history_and_rng(self):
        for solver in ('euler', 'heun'):
            for nodes in (5, 20):
                with self.subTest(solver=solver, nodes=nodes):
                    model = SmallNetwork().eval()
                    parameters = copy.deepcopy(model.state_dict())
                    inputs, labels = self.inputs()
                    wrapper = SiTDiFA(model, solver=solver, chunk_size=2)
                    raw = []
                    hook = model.register_forward_hook(lambda module, args, result: raw.append(result.detach().clone()))
                    before_rng = torch.get_rng_state().clone()
                    def observe(adapter, state, times, result):
                        stock = _cfg_output(raw[-1], 1.5)
                        self.assertTrue(torch.equal(result[3:], stock[3:]))
                        if float(times[0]) == 1.:
                            self.assertTrue(torch.equal(result, stock))
                        if adapter.pending is None:
                            expected_raw = state[:3] + (1. - float(times[0])) * stock[:3]
                            self.assertTrue(torch.equal(adapter.reference.snapshot()[-1].prediction, expected_raw))
                    try:
                        integrate(inputs, wrapper, labels, solver, nodes, observe)
                    finally:
                        hook.remove()
                    expected = (nodes - 1) * (1 if solver == 'euler' else 2)
                    self.assertEqual((model.calls, wrapper.stats['calls'], wrapper.reference.commits), (expected,) * 3)
                    self.assertEqual(wrapper.stats['intervals'], nodes - 1)
                    self.assertIsNone(wrapper.pending)
                    self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
                    self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in parameters.items()))
                    if solver == 'heun':
                        for first, second in zip(wrapper.stats['trace'][::2], wrapper.stats['trace'][1::2]):
                            self.assertEqual(first['read_history_times'], second['read_history_times'])
                            self.assertEqual(first['memory_mass_read'], second['memory_mass_read'])
                            self.assertEqual(first['difa_memory_mass_read'], second['difa_memory_mass_read'])

    def test_disabled_refinement_matches_original_cfg_trajectory(self):
        for solver in ('euler', 'heun'):
            model = SmallNetwork().eval()
            inputs, labels = self.inputs()
            wrapped, _ = integrate(inputs, SiTDiFA(model, solver=solver, guide_scale=1.), labels, solver, 5)
            stock, _ = integrate(inputs, model.forward_with_cfg, labels, solver, 5)
            self.assertTrue(torch.equal(wrapped, stock))

    def test_chunking_and_reset(self):
        model = SmallNetwork().eval()
        inputs, labels = self.inputs()
        small = SiTDiFA(model, chunk_size=1)
        large = SiTDiFA(model, chunk_size=3)
        first, _ = integrate(inputs, small, labels, 'heun', 5)
        second, _ = integrate(inputs, large, labels, 'heun', 5)
        self.assertTrue(torch.equal(first, second))
        small.reset()
        repeated, _ = integrate(inputs, small, labels, 'heun', 5)
        self.assertTrue(torch.equal(first, repeated))

    def test_four_memory_commit_failure_rolls_back(self):
        wrapper = SiTDiFA(SmallNetwork().eval())
        records, joint, peers, response = [], [], [], []
        for index in range(4):
            value = torch.randn(2, 4, 8, 8)
            conditional, base = value + .1, value - .1
            records.append(wrapper.reference.record(value, index - 3., index / 5.))
            joint.append(standardize_components(value))
            peers.append((standardize_components(conditional), standardize_components(base)))
            response.append(response_components(conditional - base))
        wrapper._commit_views(records[:2], joint[:2], peers[:2], response[:2])
        memories = (wrapper.memory_state, *wrapper.peer_memory_states, wrapper.response_memory_state)
        snapshots = copy.deepcopy(tuple(state.snapshot() for state in memories))
        old_history = copy.deepcopy(wrapper.reference.snapshot())
        old_commits, old_time = wrapper.reference.commits, wrapper.memory_last_time
        original = wrapper.response_memory_state.commit
        def fail_after_mutation(level, observed):
            original(level, observed)
            raise RuntimeError('injected commit failure')
        with patch.object(wrapper.response_memory_state, 'commit', side_effect=fail_after_mutation):
            with self.assertRaisesRegex(RuntimeError, 'injected commit failure'):
                wrapper._commit_views(records[2:], joint[2:], peers[2:], response[2:])
        self.assertTrue(equal_state(snapshots, tuple(state.snapshot() for state in memories)))
        self.assertTrue(equal_state(old_history, wrapper.reference.snapshot()))
        self.assertEqual((old_commits, old_time), (wrapper.reference.commits, wrapper.memory_last_time))
        wrapper._commit_views(records[2:], joint[2:], peers[2:], response[2:])
        self.assertEqual(wrapper.reference.commits, 4)

    def test_cli_budgets_seed_and_native_cfg(self):
        for solver in ('euler', 'heun'):
            multiplier = 1 if solver == 'euler' else 2
            for nodes in (5, 20):
                parser = create_parser()
                args = resolve_sampling(parser.parse_args(['--solver', solver, '--num-steps', str(multiplier * (nodes - 1))]), parser)
                self.assertEqual((args.actual_nfe, args.seed), (multiplier * (nodes - 1), 1))
                self.assertEqual((args.num_steps, args.time_nodes), (args.actual_nfe, nodes))
                same = resolve_sampling(parser.parse_args(['--solver', solver, '--num-evaluations', str(args.actual_nfe)]), parser)
                self.assertEqual((same.num_steps, same.time_nodes), (args.actual_nfe, nodes))
        parser = create_parser()
        with self.assertRaises(SystemExit):
            resolve_sampling(parser.parse_args(['--solver', 'heun', '--num-evaluations', '5']), parser)
        with self.assertRaises(SystemExit):
            resolve_sampling(parser.parse_args(['--num-steps', '0']), parser)
        for budget in (1, 5, 20):
            args = resolve_sampling(parser.parse_args(['--solver', 'euler', '--num-steps', str(budget)]), parser)
            self.assertEqual((args.num_steps, args.actual_nfe, args.time_nodes), (budget, budget, budget + 1))
        args = resolve_sampling(parser.parse_args(['--solver', 'heun', '--num-steps', '20']), parser)
        self.assertEqual((args.num_steps, args.actual_nfe, args.time_nodes), (20, 20, 11))
        model = SmallNetwork().eval()
        args = resolve_sampling(parser.parse_args(['--no-difa']), parser)
        self.assertEqual(build_prediction(model, args), model.forward_with_cfg)
        self.assertTrue(create_parser().parse_args([]).difa)
        parser = create_parser({'difa': False, 'solver': 'euler'})
        self.assertEqual((parser.parse_args([]).difa, parser.parse_args([]).solver), (False, 'euler'))
        args = parser.parse_args(['--difa', '--solver', 'heun'])
        self.assertEqual((args.difa, args.solver), (True, 'heun'))
        self.assertFalse(create_parser({'difa': True}).parse_args(['--no-difa']).difa)


if __name__ == '__main__':
    unittest.main()
