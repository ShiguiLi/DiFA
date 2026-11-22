"""Check prediction-coordinate equivalence without pretrained model calls."""
import math
import unittest

import torch

from difa import DiFA, LogSNRMemory, PredictionHistory
from difa.memory import response_components, standardize_components


class PredictionCoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_prediction_types_produce_equivalent_refinement(self):
        torch.manual_seed(2)
        for channels in (3, 4):
            states = []
            modes = ('data', 'noise', 'score', 'vp_velocity', 'edm_network')
            for _ in modes:
                states.append((DiFA(), PredictionHistory(), [LogSNRMemory() for _ in range(4)]))
            grid = torch.linspace(-1, 1, 64, dtype=torch.float64).reshape(1, 1, 8, 8)
            noise = torch.randn(2, channels, 8, 8, dtype=torch.float64)
            for index, level in enumerate((-2., -1., 0.)):
                alpha = math.sqrt(1. / (1. + math.exp(-level)))
                sigma = math.sqrt(1. - alpha * alpha)
                base = noise * 0.4 + index * 0.03
                conditional = base + (0.1 + index * 0.02) * torch.sin(grid * (3 + index))
                prediction = base + 1.5 * (conditional - base)
                xt = alpha * noise + sigma * noise.flip(-1)
                outputs = []
                for mode, (algorithm, history, memories) in zip(modes, states):
                    def convert(clean):
                        eps = (xt - alpha * clean) / sigma
                        if mode == 'data': return clean
                        if mode == 'noise': return (xt - sigma * eps) / alpha
                        if mode == 'score': return (xt + sigma ** 2 * (-eps / sigma)) / alpha
                        if mode == 'vp_velocity': return alpha * xt - sigma * (alpha * eps - sigma * clean)
                        # EDM preconditioning must be composed around its network output.
                        skip, out = 0.3, 0.7
                        network_output = (clean - skip * xt) / out
                        return skip * xt + out * network_output
                    current = convert(prediction)
                    cond, null = convert(conditional), convert(base)
                    torch.testing.assert_close(current, prediction, atol=1e-14, rtol=1e-14)
                    joint, conditional_memory, base_memory, response_memory = memories
                    result = algorithm.refine(current, history.snapshot(), log_snr=level,
                        high_frequency_weight=0.5, memory=joint.moments, mass=joint.mass,
                        peers=(cond, null), peer_memory=(conditional_memory.moments, base_memory.moments),
                        peer_mass=conditional_memory.mass, mixing=current.new_full((1, channels, 1, 1), 1.5),
                        response_memory=response_memory.moments, response_mass=response_memory.mass)
                    outputs.append(result.correction)
                    if index == 2:
                        self.assertEqual(result.diagnostics['difa_exact_branch_fallback'], 0.)
                    joint.commit(level, standardize_components(current))
                    conditional_memory.commit(level, standardize_components(cond))
                    base_memory.commit(level, standardize_components(null))
                    response_memory.commit(level, response_components(cond - null))
                    history.commit(history.record(current, level))
                for output in outputs[1:]:
                    torch.testing.assert_close(output, outputs[0], atol=2e-12, rtol=2e-12)

    def test_affine_flow_and_sit_conversion(self):
        torch.manual_seed(2)
        clean = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        eps = torch.randn_like(clean)
        alpha, sigma, alpha_dot, sigma_dot = 0.7, 0.4, 0.2, -0.3
        xt = alpha * clean + sigma * eps
        velocity = alpha_dot * clean + sigma_dot * eps
        recovered = (sigma_dot * xt - sigma * velocity) / (alpha * sigma_dot - sigma * alpha_dot)
        torch.testing.assert_close(recovered, clean, atol=1e-14, rtol=1e-14)
        for t in (0., 0.25, 0.75):
            xt = t * clean + (1. - t) * eps
            velocity = clean - eps
            torch.testing.assert_close(xt + (1. - t) * velocity, clean, atol=1e-14, rtol=1e-14)

    def test_spatial_layout_requirements_are_explicit(self):
        algorithm = DiFA()
        with self.assertRaises(ValueError):
            algorithm.refine(torch.ones(2, 4, 8), log_snr=0., high_frequency_weight=0.5)
        with self.assertRaises(ValueError):
            algorithm.refine(torch.ones(2, 4, 7, 8), log_snr=0., high_frequency_weight=0.5)


if __name__ == '__main__':
    unittest.main()
