#!/usr/bin/env python3
"""Prepare the official ADM evaluator and its MIT license."""
import argparse
from pathlib import Path
import shutil
from urllib.request import urlopen

UPSTREAM = 'https://raw.githubusercontent.com/openai/guided-diffusion/main/'
FILES = {'evaluator.py': 'evaluations/evaluator.py', 'LICENSE': 'LICENSE'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source-dir', type=Path, default=None, help='Existing guided-diffusion checkout; omit to download the official source')
    parser.add_argument('--output-dir', type=Path, default=Path('./third_party/adm'), help='Evaluator and license destination')
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, relative in FILES.items():
        destination = args.output_dir / name
        if args.source_dir is not None:
            shutil.copyfile(args.source_dir / relative, destination)
        else:
            with urlopen(UPSTREAM + relative, timeout=60) as response:
                destination.write_bytes(response.read())
    print('ADM evaluator available in ' + str(args.output_dir))


if __name__ == '__main__':
    main()
