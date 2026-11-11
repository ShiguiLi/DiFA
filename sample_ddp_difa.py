# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate PNG samples and a streaming NPZ with fixed DiFA G03 inference.

Run with torchrun. The measured 50,000-image profile uses two GPUs,
global seed 1 and 3,125 samples per rank per batch.
"""
import argparse
import math
import os
from pathlib import Path
import zipfile

from sample_difa import DEFAULT_CHECKPOINT, add_difa_arguments, positive_int, vae_source


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """Write arr_0 one PNG at a time instead of holding 50k images in RAM."""
    import numpy as np
    from PIL import Image

    if num <= 0:
        raise ValueError("num must be positive")
    folder = Path(sample_dir)
    target = Path(str(folder) + ".npz")
    partial = Path(str(target) + ".partial")
    shape = None
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as archive:
        with archive.open("arr_0.npy", "w", force_zip64=True) as stream:
            for index in range(num):
                with Image.open(folder / f"{index:06d}.png") as image:
                    frame = np.asarray(image, dtype=np.uint8)
                if shape is None:
                    shape = frame.shape
                    if len(shape) != 3 or shape[-1] != 3:
                        raise ValueError("Expected RGB PNG samples")
                    np.lib.format.write_array_header_1_0(stream, {
                        "descr": "|u1", "fortran_order": False,
                        "shape": (num, *shape),
                    })
                if frame.shape != shape:
                    raise ValueError(f"Inconsistent PNG shape at sample {index}")
                stream.write(frame.tobytes())
                if (index + 1) % 5000 == 0:
                    print(f"Building NPZ: {index + 1}/{num}", flush=True)
    os.replace(partial, target)
    print(f"Saved {target} [shape={(num, *shape)}].", flush=True)
    return str(target)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_difa_arguments(parser)
    parser.add_argument("--global-seed", "--seed", dest="global_seed", type=int, default=1,
                        help="Rank seed = global seed * world size + rank (two GPUs: 2, 3)")
    parser.add_argument("--per-proc-batch-size", type=positive_int, default=3125,
                        help="Samples per GPU; changing batch size changes the random sample stream")
    parser.add_argument("--num-fid-samples", "--num-samples", dest="num_fid_samples",
                        type=positive_int, default=50_000,
                        help="Requested NPZ image count; no evaluation is run")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/difa_g03"),
                        help="PNG output directory; NPZ is saved beside it as <directory>.npz")
    parser.add_argument("--skip-npz", action="store_true", help="Save only PNG samples")
    return parser


def run(args):
    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKL
    from download import find_model
    from models import SiT_models
    from transport import create_transport, Sampler
    from difa import G03Guidance
    from PIL import Image
    from tqdm import tqdm

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed sampling requires CUDA; sample_difa.py supports CPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        seed = args.global_seed * world + rank
        torch.manual_seed(seed)
        device = torch.device("cuda", local_rank)
        print(f"Starting rank={rank}, seed={seed}, world_size={world}.", flush=True)

        model = SiT_models[args.model](input_size=32, num_classes=1000,
                                       learn_sigma=True).to(device)
        checkpoint = args.ckpt or DEFAULT_CHECKPOINT
        # The automatic download writes a shared path; let rank zero finish
        # it before peers read it. Local checkpoint loading needs no barrier.
        if args.ckpt is not None or rank == 0:
            state_dict = find_model(checkpoint)
        if args.ckpt is None:
            dist.barrier()
            if rank != 0:
                state_dict = find_model(checkpoint)
        model.load_state_dict(state_dict)
        del state_dict
        model.eval()
        sampler = Sampler(create_transport("Linear", "velocity", None, None, None))
        sample = sampler.sample_ode(sampling_method="heun2", num_steps=20,
                                    atol=1e-6, rtol=1e-3, reverse=False)
        vae = AutoencoderKL.from_pretrained(vae_source(args.vae)).to(device).eval()
        vae.enable_slicing()

        folder = args.sample_dir
        if rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Saving PNG samples at {folder}", flush=True)
        dist.barrier()
        n = args.per_proc_batch_size
        total = math.ceil(args.num_fid_samples / (n * world)) * n * world
        batches = total // (n * world)
        if rank == 0:
            print(f"Generating {total} images; NPZ contains first {args.num_fid_samples}. "
                  f"Batch/rank={n}; 20 time nodes / 38 model calls per batch.", flush=True)
        for batch in tqdm(range(batches), disable=rank != 0):
            # Preserve the measured order: draw all noise, then random labels.
            z = torch.randn(n, 4, 32, 32, device=device)
            labels = torch.randint(0, 1000, (n,), device=device)
            z = torch.cat([z, z], 0)
            labels = torch.cat([labels, torch.full((n,), 1000, device=device)], 0)
            guidance = G03Guidance(model)
            latent = sample(z, guidance, y=labels, cfg_scale=1.5)[-1]
            del guidance
            latent = latent.chunk(2, dim=0)[0]
            images = vae.decode(latent / 0.18215).sample
            images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(
                0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for index, image in enumerate(images):
                image_id = batch * n * world + rank * n + index
                Image.fromarray(image).save(folder / f"{image_id:06d}.png")
            del images, latent, z, labels
            dist.barrier()

        if rank == 0:
            if not args.skip_npz:
                create_npz_from_sample_folder(folder, args.num_fid_samples)
            print("Done.", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main(argv=None):
    run(create_parser().parse_args(argv))


if __name__ == "__main__":
    main()
