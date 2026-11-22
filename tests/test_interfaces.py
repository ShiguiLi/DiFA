"""CPU checks for public entry points and external-backend invocation."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from difa.io import pack_samples
from difa.sampling import run
from difa.adapters.sit import SiTDiFA, _cfg_output
from sample import create_parser, resolve_sampling

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmallBackbone(torch.nn.Module):
    """Small nonlinear substitute; no pretrained backbone is loaded."""
    patch_size = 2

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.calls = 0
        self.native_cfg_calls = 0
        self.initial_evaluations = []

    def forward(self, x, t, y):
        self.calls += 1
        if float(t[0]) == 0.:
            self.initial_evaluations.append((x.detach().clone(), y.detach().clone()))
        px = torch.linspace(-1., 1., x.shape[-1], device=x.device)[None, None, None]
        py = torch.linspace(-1., 1., x.shape[-2], device=x.device)[None, None, :, None]
        channels = torch.arange(4, device=x.device)[None, :, None, None] * .27
        response = torch.tanh(.3 * x + torch.sin(px * 2.1 + py * 1.7 + channels))
        label = y[:, None, None, None].to(x) / 1000. - .5
        return (self.weight * torch.tanh(x) + .08 * torch.sin(t[:, None, None, None] * 2.3)
                + .18 * label * response)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        self.native_cfg_calls += 1
        n = len(x) // 2
        return _cfg_output(self(torch.cat([x[:n], x[:n]]), t, y), cfg_scale)


class SmallVAE:
    """Deterministic decoder substitute for entry-point tests."""
    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert local_files_only
        return cls()

    def to(self, device): return self
    def eval(self): return self
    def enable_slicing(self): pass
    def decode(self, latents):
        return types.SimpleNamespace(sample=F.interpolate(torch.tanh(latents[:, :3]), size=(256, 256)))


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_streaming_rgb_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'samples'
            folder.mkdir()
            frames = [np.full((8, 8, 3), value, np.uint8) for value in (0, 71)]
            for index, frame in enumerate(frames):
                Image.fromarray(frame).save(folder / f'{index:06d}.png')
            packed = pack_samples(folder, 2)
            with np.load(packed) as data:
                np.testing.assert_array_equal(data['arr_0'], np.stack(frames))
            Image.new('L', (8, 8)).save(folder / '000000.png')
            with self.assertRaisesRegex(ValueError, 'RGB'):
                pack_samples(folder, 1)

    def test_evaluation_invokes_cpu_backend_and_records_all_metrics(self):
        evaluator = load_script('difa_eval_entry', ROOT / 'eval.py')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            samples, reference, graph = (folder / name for name in ('samples.npz', 'reference.npz', 'classify_image_graph_def.pb'))
            for path in (samples, reference, graph): path.touch()
            backend = folder / 'fixture_evaluator.py'
            backend.write_text(
                "import os, pathlib, sys\n"
                "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
                "assert pathlib.Path('classify_image_graph_def.pb').is_file()\n"
                "assert len(sys.argv) == 3 and all(pathlib.Path(p).is_file() for p in sys.argv[1:])\n"
                "print('Inception Score: 12.5\\nFID: 2.1\\nsFID: 3.2\\nPrecision: 0.7\\nRecall: 0.6')\n",
                encoding='utf-8')
            args = evaluator.create_parser().parse_args([
                '--samples-npz', str(samples), '--reference-statistics', str(reference),
                '--inception-graph', str(graph), '--evaluator-script', str(backend),
                '--output-file', str(folder / 'metrics.json')])
            with contextlib.redirect_stdout(io.StringIO()): evaluator.run(args)
            record = json.loads(args.output_file.read_text(encoding='utf-8'))
            self.assertEqual(record['metrics'], {'IS': 12.5, 'FID': 2.1, 'sFID': 3.2, 'Precision': 0.7, 'Recall': 0.6})
            self.assertEqual(record['evaluation_device'], 'CPU')
            backend.write_text("print('FID: 2.1')\n", encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'five expected metrics'):
                evaluator.run(args)

    def test_evaluation_backend_local_setup(self):
        setup = load_script('difa_eval_setup', ROOT / 'scripts/setup_evaluation.py')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            (source / 'evaluations').mkdir(parents=True)
            (source / 'evaluations/evaluator.py').write_text('# evaluator fixture\n', encoding='utf-8')
            (source / 'LICENSE').write_text('license fixture\n', encoding='utf-8')
            destination = Path(tmp) / 'backend'
            with contextlib.redirect_stdout(io.StringIO()):
                setup.main(['--source-dir', str(source), '--output-dir', str(destination)])
            self.assertEqual((destination / 'evaluator.py').read_bytes(), (source / 'evaluations/evaluator.py').read_bytes())
            self.assertEqual((destination / 'LICENSE').read_bytes(), (source / 'LICENSE').read_bytes())

    def test_sampling_entry_with_real_solver_and_small_backbone(self):
        from SiT.transport import Sampler, create_transport

        models = types.ModuleType('SiT.models')
        made = []
        def factory(**kwargs):
            model = SmallBackbone()
            made.append(model)
            return model
        models.SiT_models = {'SiT-XL/2': factory}
        diffusers = types.ModuleType('diffusers')
        diffusers_models = types.ModuleType('diffusers.models')
        decoded = []
        class RecordingVAE(SmallVAE):
            def decode(self, latents):
                decoded.append(latents.detach().clone())
                return super().decode(latents)
        diffusers_models.AutoencoderKL = RecordingVAE
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {
                'SiT.models': models, 'diffusers': diffusers, 'diffusers.models': diffusers_models}), \
                patch.dict(os.environ, {}, clear=True):
            checkpoint = Path(tmp) / 'model.pt'
            torch.save(SmallBackbone().state_dict(), checkpoint)
            vae = Path(tmp) / 'vae'
            vae.mkdir()
            (vae / 'config.json').write_text('{}', encoding='utf-8')
            for solver in ('euler', 'heun'):
                baseline_inputs = baseline_latents = None
                for use_difa in (False, True):
                    with self.subTest(solver=solver, difa=use_difa):
                        parser = create_parser()
                        args = resolve_sampling(parser.parse_args([
                            '--checkpoint', str(checkpoint), '--vae', str(vae), '--device', 'cpu',
                            '--difa' if use_difa else '--no-difa', '--solver', solver,
                            '--num-steps', '5' if solver == 'euler' else '20',
                            '--batch-size', '2', '--num-samples', '3',
                            '--output-dir', str(Path(tmp) / f'{solver}_{use_difa}')]), parser)
                        decoded.clear()
                        adapters = []
                        def adapter_factory(*positional, **keywords):
                            adapter = SiTDiFA(*positional, **keywords)
                            adapters.append(adapter)
                            return adapter
                        was_enabled = torch.is_grad_enabled()
                        output = io.StringIO()
                        try:
                            with patch('difa.adapters.SiTDiFA', side_effect=adapter_factory), \
                                    contextlib.redirect_stdout(output):
                                run(args)
                        finally: torch.set_grad_enabled(was_enabled)
                        model = made[-1]
                        self.assertEqual(model.calls, 2 * args.actual_nfe)
                        self.assertEqual(model.native_cfg_calls, 0 if use_difa else model.calls)
                        self.assertEqual(len(adapters), 2 if use_difa else 0)
                        for adapter in adapters:
                            self.assertEqual(adapter.stats['calls'], args.actual_nfe)
                            self.assertIsNone(adapter.pending)
                        self.assertIn(f'solver: {solver}', output.getvalue())
                        self.assertIn('Sampling mode: DiFA' if use_difa else 'DiFA disabled', output.getvalue())
                        self.assertEqual(len(list(args.output_dir.glob('*.png'))), 3)
                        with np.load(str(args.output_dir) + '.npz') as data:
                            self.assertEqual((data['arr_0'].shape, data['arr_0'].dtype),
                                             ((3, 256, 256, 3), np.dtype('uint8')))
                        metadata = json.loads((args.output_dir / 'run_config.json').read_text(encoding='utf-8'))
                        self.assertEqual((metadata['seed'], metadata['world_size'],
                                          metadata['model_evaluations_per_batch']), (1, 1, args.actual_nfe))
                        self.assertEqual((metadata['difa'], metadata['solver']), (use_difa, solver))
                        if not use_difa:
                            baseline_inputs = model.initial_evaluations
                            baseline_latents = list(decoded)
                            native = SmallBackbone().eval()
                            native_sampler = Sampler(create_transport('Linear', 'velocity', None, None, None))
                            native_sample = native_sampler.sample_ode(
                                sampling_method=args.sampling_method, num_steps=args.time_nodes)
                            with torch.no_grad():
                                for (noise, conditions), observed in zip(baseline_inputs, baseline_latents):
                                    expected = native_sample(noise, native.forward_with_cfg,
                                        y=conditions, cfg_scale=args.cfg_scale)[-1][:2] / 0.18215
                                    self.assertTrue(torch.equal(observed, expected))
                        else:
                            for initial, baseline in zip(model.initial_evaluations, baseline_inputs):
                                self.assertTrue(all(torch.equal(a, b) for a, b in zip(initial, baseline)))
                            self.assertGreater(max(float((a - b).abs().max())
                                for a, b in zip(decoded, baseline_latents)), 1e-6)


if __name__ == '__main__':
    unittest.main()
