# SPDX-License-Identifier: MIT
"""Single-process and distributed SiT sampling, with optional DiFA refinement."""
import json
import math
import os
from pathlib import Path


def build_prediction(model, args):
    if not args.difa:
        return model.forward_with_cfg
    from .adapters import SiTDiFA
    return SiTDiFA(model, guide_scale=args.guidance_scale, window_size=args.history_size,
                   chunk_size=args.refinement_chunk_size, solver=args.solver)


def run(args):
    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKL
    from PIL import Image
    from SiT.models import SiT_models
    from SiT.transport import Sampler, create_transport
    from .io import pack_samples

    distributed = 'RANK' in os.environ
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('torchrun sampling requires CUDA and NCCL')
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device('cuda', local_rank)
    else:
        rank, world = 0, 1
        selected = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(selected if args.device == 'auto' else args.device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        rank_seed = args.seed * world + rank
        torch.manual_seed(rank_seed)
        if rank == 0:
            mode = 'DiFA' if args.difa else 'Original model (DiFA disabled)'
            print(f'Sampling mode: {mode}; solver: {args.solver}; '
                  f'steps: {args.num_steps}; seed: {args.seed}', flush=True)
        if not args.checkpoint.is_file():
            raise FileNotFoundError('Checkpoint not found: ' + str(args.checkpoint))
        if not (args.vae / 'config.json').is_file():
            raise FileNotFoundError('VAE config.json not found in ' + str(args.vae))
        model = SiT_models[args.model](input_size=32, num_classes=1000, learn_sigma=True).to(device)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'ema' in checkpoint:
            checkpoint = checkpoint['ema']
        model.load_state_dict(checkpoint)
        del checkpoint
        model.eval()
        sampler = Sampler(create_transport('Linear', 'velocity', None, None, None))
        sample = sampler.sample_ode(sampling_method=args.sampling_method, num_steps=args.time_nodes,
                                    atol=1e-6, rtol=1e-3, reverse=False)
        vae = AutoencoderKL.from_pretrained(args.vae, local_files_only=True).to(device).eval()
        vae.enable_slicing()
        folder = args.output_dir
        if rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            metadata.update(world_size=world, model_evaluations_per_batch=args.actual_nfe)
            (folder / 'run_config.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        if distributed: dist.barrier()
        n = args.batch_size
        total = math.ceil(args.num_samples / (n * world)) * n * world
        for batch in range(total // (n * world)):
            # Preserve the measured random stream: noise, then class labels.
            noise = torch.randn(n, 4, 32, 32, device=device)
            if args.labels is None:
                labels = torch.randint(0, 1000, (n,), device=device)
            else:
                labels = torch.tensor([args.labels[(batch * n * world + rank * n + index) % len(args.labels)]
                                       for index in range(n)], device=device)
            inputs = torch.cat([noise, noise], dim=0)
            conditions = torch.cat([labels, torch.full_like(labels, 1000)], dim=0)
            prediction = build_prediction(model, args)
            latents = sample(inputs, prediction, y=conditions, cfg_scale=args.cfg_scale)[-1]
            del prediction
            latents = latents.chunk(2, dim=0)[0]
            decoded = vae.decode(latents / 0.18215).sample
            images = torch.clamp(127.5 * decoded + 128., 0., 255.).permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8).numpy()
            for index, image in enumerate(images):
                image_id = batch * n * world + rank * n + index
                if image_id < args.num_samples:
                    Image.fromarray(image).save(folder / f'{image_id:06d}.png')
            if args.save_grid and rank == 0 and batch == 0:
                from torchvision.utils import save_image
                save_image(decoded, folder / 'grid.png', nrow=4, normalize=True, value_range=(-1, 1))
            if distributed: dist.barrier()
            if rank == 0:
                print(f'Batch {batch + 1}/{total // (n * world)}; {args.num_steps} sampling steps', flush=True)
            del decoded, images, latents, inputs, conditions, noise, labels
        if rank == 0 and not args.skip_packing:
            print('Saved ' + str(pack_samples(folder, args.num_samples)), flush=True)
        if distributed: dist.barrier()
    finally:
        if distributed: dist.destroy_process_group()
