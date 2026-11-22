#!/usr/bin/env python3
"""Sample pretrained SiT-XL/2 with original predictions or DiFA refinement."""
import argparse
import json
import math
from pathlib import Path

DEFAULTS = dict(model='SiT-XL/2', image_size=256, num_classes=1000,
                checkpoint='./data/pretrained/SiT-XL-2-256x256.pt',
                vae='./data/pretrained/sd-vae-ft-ema', solver='heun', num_steps=20,
                cfg_scale=1.5, guidance_scale=1.75, difa=True, seed=1, batch_size=32,
                num_samples=8, history_size=3, refinement_chunk_size=125,
                output_dir='./outputs/samples')


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError('must be finite')
    return number


def create_parser(defaults=None):
    values = dict(DEFAULTS, **(defaults or {}))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path('configs/imagenet256.json'),
                        help='JSON configuration; explicit command-line options override it')
    parser.add_argument('--model', choices=['SiT-XL/2'], default=values['model'], help='Pretrained backbone')
    parser.add_argument('--image-size', type=int, choices=[256], default=values['image_size'], help='Output resolution')
    parser.add_argument('--num-classes', type=int, choices=[1000], default=values['num_classes'], help='ImageNet class count')
    parser.add_argument('--checkpoint', type=Path, default=Path(values['checkpoint']), help='Local SiT checkpoint')
    parser.add_argument('--vae', type=Path, default=Path(values['vae']), help='Local EMA VAE directory containing config.json')
    parser.add_argument('--solver', choices=['euler', 'heun'], default=values['solver'], help='Sampling solver for both original-model and DiFA modes')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--num-steps', type=positive_int, default=values['num_steps'], help='Number of sampling steps; Heun requires an even count')
    budget.add_argument('--num-evaluations', type=positive_int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--cfg-scale', type=finite_float, default=values['cfg_scale'], help='Classifier-free guidance scale')
    parser.add_argument('--guidance-scale', type=finite_float, default=values['guidance_scale'], help='DiFA strength; values at most one disable refinement')
    parser.add_argument('--difa', action=argparse.BooleanOptionalAction, default=values['difa'], help='Enable DiFA refinement; --no-difa passes original model CFG predictions directly to the solver')
    parser.add_argument('--seed', type=int, default=values['seed'], help='Global seed; rank seed is seed * world_size + rank')
    parser.add_argument('--batch-size', type=positive_int, default=values['batch_size'], help='Images per GPU per batch')
    parser.add_argument('--num-samples', type=positive_int, default=values['num_samples'], help='Number of images in the final dataset')
    parser.add_argument('--labels', type=int, nargs='+', default=None, help='Optional ImageNet class IDs; otherwise draw random classes')
    parser.add_argument('--history-size', type=positive_int, default=values['history_size'], help='Bounded raw-prediction history size')
    parser.add_argument('--refinement-chunk-size', type=positive_int, default=values['refinement_chunk_size'], help='Tensor batch size for DiFA arithmetic, independent of model batch size')
    parser.add_argument('--output-dir', type=Path, default=Path(values['output_dir']), help='PNG output directory; NPZ is written beside it')
    parser.add_argument('--skip-packing', action='store_true', help='Save PNGs without CPU NPZ packing; pack later with eval.py --pack-only')
    parser.add_argument('--save-grid', action='store_true', help='Also save a grid of the first batch in the output directory')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='Single-process device; torchrun uses local CUDA devices')
    return parser


def resolve_sampling(args, parser):
    count = 1 if args.solver == 'euler' else 2
    if args.num_evaluations is not None:
        args.num_steps = args.num_evaluations
    if args.num_steps < 1:
        parser.error('--num-steps must be positive')
    if args.num_steps % count:
        parser.error('Heun requires an even number of sampling steps')
    if args.labels is not None and any(label < 0 or label >= 1000 for label in args.labels):
        parser.error('--labels must contain ImageNet class IDs in [0, 999]')
    args.actual_nfe = args.num_steps
    args.time_nodes = args.actual_nfe // count + 1
    args.sampling_method = 'euler' if args.solver == 'euler' else 'heun2'
    return args


def main(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', type=Path, default=Path('configs/imagenet256.json'))
    known, _ = preliminary.parse_known_args(argv)
    values = {}
    if known.config.is_file():
        values = json.loads(known.config.read_text(encoding='utf-8'))
    elif argv is not None or not any(value in {'-h', '--help'} for value in __import__('sys').argv[1:]):
        preliminary.error('Configuration not found: ' + str(known.config))
    parser = create_parser(values)
    args = resolve_sampling(parser.parse_args(argv), parser)
    from difa.sampling import run
    run(args)


if __name__ == '__main__':
    main()
