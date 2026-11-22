#!/usr/bin/env python3
"""Prepare the official SiT-XL/2 checkpoint and EMA VAE."""
import argparse
from pathlib import Path

CHECKPOINT_URL = 'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=Path('./data/pretrained'), help='Checkpoint and VAE destination')
    parser.add_argument('--skip-vae', action='store_true', help='Download only the SiT checkpoint')
    args = parser.parse_args()
    from torchvision.datasets.utils import download_url
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = 'SiT-XL-2-256x256.pt'
    if not (args.output_dir / filename).is_file():
        download_url(CHECKPOINT_URL, str(args.output_dir), filename=filename)
    if not args.skip_vae:
        from diffusers.models import AutoencoderKL
        folder = args.output_dir / 'sd-vae-ft-ema'
        local_weights = any(folder.glob('diffusion_pytorch_model*.safetensors')) or any(folder.glob('diffusion_pytorch_model*.bin'))
        if not ((folder / 'config.json').is_file() and local_weights):
            model = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
            model.save_pretrained(folder)
    print('Pretrained assets are available in ' + str(args.output_dir))


if __name__ == '__main__':
    main()
