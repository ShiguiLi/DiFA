# SiT source attribution

This directory vendors the SiT backbone (`models.py`) and its transport
package (`transport/`) for pretrained ImageNet sampling.

Upstream: [willisma/SiT](https://github.com/willisma/SiT).
Paper: [Exploring Flow and Diffusion-based Generative Models with Scalable
Interpolant Transformers](https://arxiv.org/abs/2401.08740).

The transport source includes the fixed forward Euler/Heun integration used
by this release. DiFA prediction conversion and history handling are in
[`difa/adapters/sit.py`](../difa/adapters/sit.py). The pretrained weights
remain separate downloads. See [`LICENSE.txt`](LICENSE.txt) for the
upstream MIT license and the root README for the supported commands.
