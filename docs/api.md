# Python API

The `difa` package requires PyTorch. Its public core refines spatial
data-prediction tensors `[B, C, H, W]`, independently of model and solver
imports. `DiFA()` constructs the released refinement.

Prediction-coordinate conversions are given in [Compatibility](compatibility.md).
Conversions are performed by a model adapter; the core does not infer a
model's output parameterization or choose its solver.

## Core interface

```python
from difa import DiFA, PredictionHistory, LogSNRMemory

algorithm = DiFA(guide_scale=1.75, group_size=2)
history = PredictionHistory(window_size=3)
```

`DiFA.refine()` accepts the following inputs:

| Argument | Contract |
| --- | --- |
| `prediction` | Finite floating `[B, C, H, W]` clean-data estimate; positive dimensions, with `H` and `W` divisible by the positive integer `group_size` |
| `history` | Sequence of raw `PredictionRecord` observations in the same data units, shape, dtype, device, and batch order; default empty |
| `log_snr` | Required finite scalar, conventionally `2 * log(alpha / sigma)` |
| `high_frequency_weight` | Required scalar in `[0, 1]`, supplied by the adapter |
| `memory`, `mass` | Optional two standardized-component memory moments and their finite nonnegative integration mass |
| `peers`, `mixing` | Optional `(conditional_data, base_data)` predictions and finite per-channel CFG mixing of shape `[1, C, 1, 1]` |
| `peer_memory`, `peer_mass` | Optional pair of branch memories, each containing two standardized-component moments, with a shared finite nonnegative mass |
| `response_memory`, `response_mass` | Optional two unstandardized response-component moments and their finite nonnegative mass |

All supplied prediction and memory tensors must match `prediction` in shape,
floating dtype, and device. The core returns a `Refinement` containing a
`correction` tensor with the same shape, dtype, and device, plus scalar
`diagnostics`. It does not commit observations or advance the solver.
The CFG inputs and populated memories enable the full response-statistic
term; missing views select the consensus fallback, while cold or incomplete
memory selects the documented reference fallback.

The adapter supplies:

```python
result = algorithm.refine(
    data_prediction,
    history.snapshot(),
    log_snr=log_snr,
    high_frequency_weight=residual_weight,
    memory=joint_memory.moments,
    mass=joint_memory.mass,
    peers=(conditional_data, base_data),
    peer_memory=(conditional_memory.moments, base_memory.moments),
    peer_mass=conditional_memory.mass,
    mixing=cfg_mixing,
    response_memory=response_memory.moments,
    response_mass=response_memory.mass,
)
refined_data = data_prediction + data_gain * result.correction
```

The two branch memories must have the same mass and all four memories must
advance through the same committed coordinates. `cfg_mixing` is a finite
tensor of shape `[1, C, 1, 1]`; branch predictions and memory tensors must
match the current prediction's shape, floating dtype, and device.

`Refinement.correction` is a data-derived direction `b`; the adapter chooses
the actual data-update gain and converts the result back to its native
prediction representation. It is not automatically a final-data increment.
`Refinement.diagnostics` contains scalar observations, not calibrated
confidence estimates.

## Executable tensor example

The following example demonstrates raw-memory management with synthetic
predictions and an illustrative data gain:

```python
import torch
from difa import DiFA, LogSNRMemory, PredictionHistory
from difa.memory import response_components, standardize_components

algorithm = DiFA()
history = PredictionHistory(3)
joint, conditional, base, response = [LogSNRMemory() for _ in range(4)]
grid = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
mixing = grid.new_full((1, 1, 1, 1), 1.5)

for index, level in enumerate((-2.0, -1.0, 0.0)):
    p_base = grid + 0.02 * index
    p_cond = p_base + 0.1 * torch.sin(grid * (3.0 + index))
    raw_response = p_cond - p_base
    prediction = p_base + 1.5 * raw_response
    result = algorithm.refine(
        prediction, history.snapshot(), log_snr=level,
        high_frequency_weight=0.5,
        memory=joint.moments, mass=joint.mass,
        peers=(p_cond, p_base),
        peer_memory=(conditional.moments, base.moments),
        peer_mass=conditional.mass, mixing=mixing,
        response_memory=response.moments, response_mass=response.mass,
    )
    refined = prediction + 0.5 * result.correction
    assert torch.isfinite(refined).all()

    # Commit the raw observations after the evaluation, not refined.
    joint.commit(level, standardize_components(prediction))
    conditional.commit(level, standardize_components(p_cond))
    base.commit(level, standardize_components(p_base))
    response.commit(level, response_components(raw_response))
    history.commit(history.record(prediction, level))
```

The commit timing in this example is suitable for demonstrating a single
evaluation per interval. Multi-stage or rejected-step solvers must control
snapshots and commits according to their own lifecycle.

## Memory and history

`PredictionHistory.record(prediction, log_snr, time=None)` creates a detached
raw record without committing it. `snapshot()` returns a tuple of committed
records; `commit(*records)` appends them; `reset()` clears the history.

`LogSNRMemory.commit(level, components)` accepts two finite BCHW components.
`snapshot()` and `restore(snapshot)` allow transactional updates. Snapshot
tensors must be treated as read-only. Committed log-SNR must be nondecreasing;
restore state before retrying an earlier coordinate.

## SiT model loading

Run from the repository root after installing the package and downloading the assets:

```python
import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from difa.adapters import SiTDiFA
from SiT.models import SiT_models
from SiT.transport import Sampler, create_transport

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
torch.backends.cuda.matmul.allow_tf32 = True
model = SiT_models["SiT-XL/2"](
    input_size=32, num_classes=1000, learn_sigma=True
).to(device).eval()
checkpoint = torch.load("data/pretrained/SiT-XL-2-256x256.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint.get("ema", checkpoint))
vae = AutoencoderKL.from_pretrained("data/pretrained/sd-vae-ft-ema").to(device).eval()
vae.enable_slicing()

sampler = Sampler(create_transport("Linear", "velocity", None, None, None))
sampling_steps = 20
solver = "heun"  # Select "euler" or "heun".
use_difa = True  # Set False to use the original model predictions.
if solver == "heun" and sampling_steps % 2:
    raise ValueError("Heun requires an even number of sampling steps")
nodes = sampling_steps // (2 if solver == "heun" else 1) + 1
sample = sampler.sample_ode(
    sampling_method="heun2" if solver == "heun" else "euler", num_steps=nodes
)
prediction = (
    SiTDiFA(model, solver=solver, guide_scale=1.75)
    if use_difa else model.forward_with_cfg
)
labels = torch.tensor([207, 360], device=device)
noise = torch.randn(len(labels), 4, 32, 32, device=device)
inputs = torch.cat([noise, noise], dim=0)
conditions = torch.cat([labels, torch.full_like(labels, 1000)], dim=0)

with torch.no_grad():
    latents = sample(inputs, prediction, y=conditions, cfg_scale=1.5)[-1]
    latents = latents.chunk(2, dim=0)[0]
    images = vae.decode(latents / 0.18215).sample
    save_image(images, "sample.png", nrow=2, normalize=True, value_range=(-1, 1))
```

`SiTDiFA` requires an eval-mode model, synchronized forward times in `[0,1]`,
and a fixed condition/CFG scale within a trajectory. Its `solver` must be
`"euler"` or `"heun"` and must match the downstream sampler. Use a new
wrapper per independent batch or call `reset()`.

`chunk_size` controls only refinement arithmetic batching, not model batch
size. The default is 125; configure it with the `chunk_size` argument.
For unrefined sampling use `model.forward_with_cfg`, or `--no-difa` in the CLI.
