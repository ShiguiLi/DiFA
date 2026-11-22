# Third-party notices

## SiT

`SiT/models.py` and `SiT/transport/` are derived from
[willisma/SiT](https://github.com/willisma/SiT), distributed under the MIT
license. The original notice is retained in [`SiT/LICENSE.txt`](SiT/LICENSE.txt).
The SiT backbone builds on [DiT](https://github.com/facebookresearch/DiT).

## ADM evaluation

FID, sFID, Inception Score, Precision, and Recall use the evaluator from
[openai/guided-diffusion](https://github.com/openai/guided-diffusion), an
MIT-licensed project. The setup script downloads the original evaluator
and its license into `third_party/adm/`. These files are obtained separately;
see [`third_party/adm/README.md`](third_party/adm/README.md).

## Runtime libraries and models

Sampling uses PyTorch, torchvision, torchdiffeq, timm, NumPy, SciPy, Pillow,
tqdm, Hugging Face Diffusers, and Accelerate. Evaluation uses TensorFlow.
These packages retain their respective licenses. The EMA VAE is provided
by [stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema);
pretrained assets remain subject to their providers' terms.
