# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sample a grid with the fixed DiFA G03 inference configuration."""
import argparse
from pathlib import Path
from time import perf_counter


DEFAULT_LABELS = [207, 360, 387, 974, 88, 979, 417, 279]
DEFAULT_CHECKPOINT = "SiT-XL-2-256x256.pt"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_difa_arguments(parser):
    """Shared CLI options; algorithm settings stay fixed to measured G03."""
    parser.add_argument("mode", choices=["ODE"], help="Fixed forward velocity ODE")
    parser.add_argument("--model", default="SiT-XL/2", choices=["SiT-XL/2"])
    parser.add_argument("--ckpt", default=None,
                        help="Local official XL/2-256 checkpoint; omitted: official auto-download")
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema",
                        help="EMA VAE local directory or Hugging Face model ID (also accepts ema)")
    parser.add_argument("--image-size", type=int, default=256, choices=[256])
    parser.add_argument("--num-classes", type=int, default=1000, choices=[1000])
    parser.add_argument("--cfg-scale", type=float, default=1.5, choices=[1.5])
    parser.add_argument("--difa-guide-scale", type=float, default=1.75, choices=[1.75])
    parser.add_argument("--sampling-method", default="heun2", choices=["heun2"])
    parser.add_argument("--num-sampling-steps", type=int, default=20, choices=[20],
                        help="20 time nodes, 19 Heun2 intervals, 38 model calls")


def vae_source(value):
    return "stabilityai/sd-vae-ft-ema" if value == "ema" else value


def create_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Uses CUDA when available, otherwise CPU. For distributed PNG/NPZ output use sample_ddp_difa.py.")
    add_difa_arguments(parser)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--labels", type=int, nargs="+", default=DEFAULT_LABELS,
                        metavar="CLASS", help="ImageNet class IDs, each in [0, 999]")
    parser.add_argument("--output", type=Path, default=Path("sample_difa.png"),
                        help="Output image grid path")
    return parser


def run(args):
    # Keep --help usable without loading torch, timm, torchvision or diffusers.
    import torch
    from torchvision.utils import save_image
    from diffusers.models import AutoencoderKL
    from download import find_model
    from models import SiT_models
    from transport import create_transport, Sampler
    from difa import G03Guidance

    # Match the measured sampler: FP32, TF32 matmuls, no gradients; retain the
    # PyTorch default cuDNN TF32 setting.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiT_models[args.model](input_size=32, num_classes=1000,
                                   learn_sigma=True).to(device)
    model.load_state_dict(find_model(args.ckpt or DEFAULT_CHECKPOINT))
    model.eval()
    sampler = Sampler(create_transport("Linear", "velocity", None, None, None))
    sample = sampler.sample_ode(sampling_method="heun2", num_steps=20,
                                atol=1e-6, rtol=1e-3, reverse=False)
    vae = AutoencoderKL.from_pretrained(vae_source(args.vae)).to(device).eval()
    vae.enable_slicing()

    n = len(args.labels)
    z = torch.randn(n, 4, 32, 32, device=device)
    labels = torch.tensor(args.labels, device=device)
    z = torch.cat([z, z], 0)
    labels = torch.cat([labels, torch.full((n,), 1000, device=device)], 0)
    guidance = G03Guidance(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    latent = sample(z, guidance, y=labels, cfg_scale=1.5)[-1]
    # Release the trajectory's reference caches before sliced VAE decoding.
    del guidance
    latent = latent.chunk(2, dim=0)[0]
    images = vae.decode(latent / 0.18215).sample
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, str(args.output), nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved {args.output}; device={device}, seed={args.seed}, "
          f"20 time nodes / 38 model calls, {perf_counter() - start:.2f} seconds.")


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if any(label < 0 or label >= 1000 for label in args.labels):
        parser.error("--labels must contain class IDs in [0, 999]")
    run(args)


if __name__ == "__main__":
    main()
