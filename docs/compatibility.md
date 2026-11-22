# Data-prediction compatibility

## Common prediction coordinates

For a Gaussian forward process

$$x_t=\alpha_t x_0+\sigma_t\epsilon,$$

DiFA's common variable is an estimated clean signal $\widehat{x}_0$.
It can be a pixel-space image or a latent-space tensor. Conversion must use
the coefficients and preconditioning of the particular pretrained model.

| Model output | Conversion to data prediction | Conditions |
| --- | --- | --- |
| Clean-data estimate $D_\theta$ | $\widehat{x}_0=D_\theta$ | Complete denoiser output |
| Noise $\widehat\epsilon$ | $\widehat{x}_0=(x_t-\sigma_t\widehat\epsilon)/\alpha_t$ | $\alpha_t\ne0$ |
| Score $s_\theta$ | $\widehat{x}_0=(x_t+\sigma_t^2s_\theta)/\alpha_t$ | Gaussian perturbation score; $\alpha_t\ne0$ |
| VP velocity $v_\theta=\alpha_t\widehat\epsilon-\sigma_t\widehat{x}_0$ | $\widehat{x}_0=\alpha_t x_t-\sigma_t v_\theta$ | $\alpha_t^2+\sigma_t^2=1$ |
| Linear-path flow velocity $u_\theta$ | $\widehat{x}_0=x_t+(1-t)u_\theta$ | SiT path: $\alpha_t=t$, $\sigma_t=1-t$ |

VP velocity and flow velocity use different definitions. For a general
affine flow with $u_t=\dot\alpha_t x_0+\dot\sigma_t\epsilon$,

$$
\widehat{x}_0=
\frac{\dot\sigma_t x_t-\sigma_t u_\theta}
{\alpha_t\dot\sigma_t-\sigma_t\dot\alpha_t},
$$

provided the denominator is nonzero. EDM's complete preconditioned denoiser
already returns a data estimate. When wrapping its internal network, first
form $D_\theta=c_{\mathrm{skip}}x_t+c_{\mathrm{out}}F_\theta$ using the
model's preconditioning, including its network input and noise embedding.

After refinement, invert the same conversion to obtain the native model
output expected by the solver. At singular endpoints, use an explicit
endpoint rule. The SiT adapter bypasses refinement at exactly $t=1$.

## What the released core supports

The public `difa.DiFA` class has no backbone, solver, timestep count, or
network-forward input. It accepts finite floating BCHW predictions,
same-coordinate raw observations, log-SNR, and an adapter-supplied residual
weight. Spatial dimensions must be divisible by `group_size` (default 2).
The smooth/residual decomposition uses 5 × 5 spatial pooling, and alignment
statistics are computed per image and channel over the spatial dimensions.

Consequently the core supports spatial image/latent data estimates beyond
a particular network parameterization. Token sequences, vector data, and
five-dimensional videos require explicit layout handling or a different
component decomposition. A prediction conversion alone does not establish
quality across models or modalities.

The full response-statistic term requires the raw conditional and base
views already computed for CFG. A sampler exposing only a final guided
prediction cannot recover both views from that tensor alone. In the absence
of usable views or response memory, the core follows its defined consensus
and reference fallbacks. Its full CFG formula therefore depends on access
to those observations, while retaining zero additional model evaluations.

## Adapter contract

An adapter must:

1. Convert current and historical raw predictions into consistent data units.
   Preserve the model's clipping, preconditioning, and CFG channel convention.
2. Supply a compatible log-SNR coordinate that increases along committed
   reverse-process observations, and the residual-frequency weight.
3. Supply branch predictions and per-channel mixing when CFG views are available.
4. Read frozen history for stages belonging to the same solver interval and
   commit raw observations at the appropriate lifecycle boundary. Rejected
   updates must restore history and memory.
5. Apply the correct data-update gain and invert the prediction conversion.
   `Refinement.correction` is a direction; the adapter determines its gain.

For SiT, $p=x_t+(1-t)v_{\mathrm{CFG}}$ and the implementation applies
$\widetilde p=p+(1-t)b$, equivalently $\widetilde v=v_{\mathrm{CFG}}+b$.
Only the first three SiT CFG channels are mixed. The fourth-channel base is
conditional. These are SiT adapter conventions.

The supplied SiT adapter supports forward Euler and Heun. Other solver
lifecycles require corresponding adapters. [Method](method.md) and
[Python API](api.md) describe the required history and memory interfaces.

## Coordinate checks

`tests/test_prediction_coordinates.py` checks noise, score, VP-velocity,
EDM-style preconditioning, and affine-flow conversions with synthetic
predictions. It also compares the DiFA refinement obtained from equivalent
data estimates. These are numerical interface checks; the supplied complete
pretrained-model sampling integration remains SiT.
