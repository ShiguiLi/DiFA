#!/usr/bin/env python3
"""Pack samples and run the ADM FID/sFID/Inception Score evaluation on CPU."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sample-dir', type=Path, default=None, help='Optional numbered RGB PNG directory to pack')
    parser.add_argument('--samples-npz', type=Path, default=None, help='Sample archive; defaults to <sample-dir>.npz or ./outputs/samples.npz')
    parser.add_argument('--num-samples', type=int, default=50000, help='Number of PNGs to pack')
    parser.add_argument('--reference-statistics', type=Path, default=Path('./data/references/VIRTUAL_imagenet256_labeled.npz'), help='ADM-compatible ImageNet reference NPZ')
    parser.add_argument('--inception-graph', type=Path, default=Path('./data/evaluation/classify_image_graph_def.pb'), help='ADM Inception graph file')
    parser.add_argument('--evaluator-script', type=Path, default=Path('./third_party/adm/evaluator.py'), help='Official ADM evaluator Python file')
    parser.add_argument('--output-file', type=Path, default=Path('./outputs/metrics.json'), help='JSON output containing metrics and evaluation protocol')
    parser.add_argument('--pack-only', action='store_true', help='Pack PNG samples without loading the evaluator')
    return parser


def run(args):
    archive = args.samples_npz or (Path(str(args.sample_dir) + '.npz') if args.sample_dir else Path('./outputs/samples.npz'))
    if args.sample_dir is not None:
        from difa.io import pack_samples
        archive = pack_samples(args.sample_dir, args.num_samples, archive)
        print('Saved ' + str(archive), flush=True)
    if args.pack_only:
        if args.sample_dir is None:
            raise ValueError('--pack-only requires --sample-dir')
        return
    for name, path in [('samples', archive), ('reference statistics', args.reference_statistics),
                       ('Inception graph', args.inception_graph), ('ADM evaluator', args.evaluator_script)]:
        if not path.is_file():
            raise FileNotFoundError(name + ' not found: ' + str(path))
    if args.inception_graph.name != 'classify_image_graph_def.pb':
        raise ValueError('ADM expects the graph filename classify_image_graph_def.pb')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    process = subprocess.run([sys.executable, str(args.evaluator_script.resolve()),
                              str(args.reference_statistics.resolve()), str(archive.resolve())],
                             cwd=args.inception_graph.resolve().parent, env=environment,
                             capture_output=True, text=True)
    print(process.stdout, end='')
    if process.returncode:
        print(process.stderr, file=sys.stderr)
        raise RuntimeError('ADM evaluation failed with exit code ' + str(process.returncode))
    metrics = {}
    for name, value in re.findall(r'(?m)^(Inception Score|FID|sFID|Precision|Recall):\s*([-+0-9.eE]+)\s*$', process.stdout):
        metrics['IS' if name == 'Inception Score' else name] = float(value)
    if set(metrics) != {'IS', 'FID', 'sFID', 'Precision', 'Recall'}:
        raise RuntimeError('ADM output did not contain all five expected metrics')
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    record = {'metrics': metrics, 'samples_npz': str(archive),
              'reference_statistics': str(args.reference_statistics),
              'inception_graph': str(args.inception_graph), 'evaluation_device': 'CPU'}
    args.output_file.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Saved ' + str(args.output_file))


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.num_samples < 1: parser.error('--num-samples must be positive')
    run(args)


if __name__ == '__main__':
    main()
