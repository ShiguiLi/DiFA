# DiFA implementation

DiFA implements the DiFA framework as a temporal-consensus correction plus
conditional-response statistic innovation. It uses the consensus consensus main
term and preserves its cold-start and missing-view fallbacks. The equations
below describe this repository's implementation.

## Data coordinates and observations

The model adapter supplies a current CFG data prediction `p`, two raw
data views `p_cond` and `p_base`, and their response:

$$R=p_{\mathrm{cond}}-p_{\mathrm{base}}.$$

Each BCHW tensor is split into a smooth component and its residual:

$$L(x)=\operatorname{AvgPool}_{5\times5}(x),\qquad
x_0=L(x),\quad x_1=x-L(x).$$

These are complementary components, not orthogonal frequency bands.
For each image and channel, component statistics are computed over the
spatial dimensions using population standard deviation.

## Raw-history memories

Four log-SNR memories are maintained:

| Memory | Stored raw observation |
| --- | --- |
| Joint prediction | Standardized components of `p` |
| Conditional branch | Standardized components of `p_cond` |
| Base branch | Standardized components of `p_base` |
| Response | Unstandardized components of `R` |

`LogSNRMemory` uses an exponential integral with scale one log-SNR unit,
integrating piecewise-linear observations. Memory moments are divided by
their accumulated mass when read. Repeated levels update the latest
observation without adding an integration interval or mass. A separate
bounded raw-prediction history uses a window of three records by default.

The memory represents historical observations, not calibrated prediction
confidence. Conditional and base predictions have different targets; their
disagreement is not an estimate of clean-prediction error.

## Statistic innovation

Let `R_i` be the current response component and
`H_i = M_response,i / mass` its raw historical reference. Write:

$$
R_i=\mu_{c,i}+s_{c,i}z_{c,i},\qquad
H_i=\mu_{h,i}+s_{h,i}z_{h,i},
$$

where `s` is the standard deviation clamped below by `1e-6`, and
`z = (component - mean) / s`. The symmetric decomposition is:

$$
D_{\mathrm{shape}}=
\sum_i \frac{s_{c,i}+s_{h,i}}2(z_{c,i}-z_{h,i}),
$$

$$
D_{\mathrm{stat}}=
\sum_i\left[\mu_{c,i}-\mu_{h,i}
+\frac{s_{c,i}-s_{h,i}}2(z_{c,i}+z_{h,i})\right].
$$

Their sum equals `R - sum_i H_i`, up to numerical rounding. DiFA uses
`D_stat` as its additional signal. It is a spatial tensor: standardized
shapes carry the mean/scale innovation. The additional term vanishes when
the component means and scales match, even if the shapes differ.

## Complete refinement

With usable response history, DiFA forms the data-derived correction:

$$
b_{\mathrm{DiFA}}=b_{\mathrm{consensus}}
+(\gamma-1)(W-\tfrac12)D_{\mathrm{stat}}.
$$

`gamma` is `guide_scale`, and `W` is the adapter's per-channel CFG mixing
tensor. `b_consensus` retains aligned temporal consensus, reference transport,
coordinate-balanced static guidance, and damping of the supplemental static
increment along an observable repeated-coordinate disagreement direction.
Its implementation is in [consensus.py](../difa/_impl/consensus.py), with
the supporting reference/guidance operations in the same private package.

The response term is applied only for a numerically valid response axis.
The composed correction preserves prediction-add/subtract rounding.
The additional statistic term is not subjected to the main term's radial
projection or frequency filter. Shapes and statistics are not orthogonal
guidance components.

The cold path follows the inherited instantaneous two-view reference
refinement. Missing response or branch memory follows the branch/temporal-reference
fallback; missing views reduces to the consensus main term. `guide_scale <= 1`
disables refinement. These paths are part of DiFA's complete implementation.

## SiT conversion

For the forward linear SiT path, time runs from zero to one:

$$p=x_t+(1-t)v_{\mathrm{CFG}},\qquad
\ell(t)=2\log\frac{t}{1-t}.$$

The adapter clamps time to `[1e-6, 1-1e-6]` for the log-SNR calculation.
It supplies the residual-frequency weight
`1 - sigmoid(ell)` to the consensus main term. The actual data update is:

$$\widetilde p=p+(1-t)b_{\mathrm{DiFA}},\qquad
\widetilde v=v_{\mathrm{CFG}}+b_{\mathrm{DiFA}}.$$

At the exact endpoint `t=1`, refinement is bypassed. The raw CFG convention
of SiT is preserved: only its first three velocity channels are mixed;
the fourth-channel base is conditional. Consequently the added response
statistic signal is zero in that channel, while the consensus main term
can still act on it. This channel convention is specific to the SiT adapter.

## Solver lifecycle

The core has no solver-name, stage, step-size, or node-count input. Euler
commits its raw prediction after an evaluation. Heun reads one frozen
history/memory snapshot for both stages of an interval, then commits the
raw start and provisional-endpoint observations together. All four memory
updates are transactional.

Changing solvers changes the available observations. Other solvers require
an adapter with appropriate stage, cache, acceptance, and rollback handling;
core independence does not establish equal quality or convergence order
across solvers. Current quality results use uniform Heun2.
