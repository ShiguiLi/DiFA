# DiFA inference integration

Base: the complete official [SiT](https://github.com/willisma/SiT) project at
commit `cbde832a40b153ccc79603412409da9c9b0c568c`.

All upstream source, configuration, notebook, image and license files are
retained. Only the tracked macOS metadata file `visuals/.DS_Store` is omitted;
`.gitignore` adds exclusions for local weights, caches and generated output.
Root README.md introduces DiFA; the original upstream README is preserved
as README_SiT.md. The original
models.py, train.py, train_utils.py, sample.py, sample_ddp.py, download.py,
environment.yml, transport/, notebook, visual images and LICENSE.txt are unchanged.

Added inference files:

- `difa/`: measured G03 reference/guidance implementation and inference adapter.
- `sample_difa.py`: single-process generation through the existing SiT sampler.
- `sample_ddp_difa.py`: distributed generation through the existing SiT sampler.
- `requirements-difa.txt`: inference dependencies.

`difa/adapters/observer.py` temporarily observes the last residual block's input,
norm2 input (the state after attention), output and final readout inputs. It
does not modify model outputs, parameters or layer definitions. Hook handles
are removed at the end of every call. This supplies the same tensors used by
the previously measured observer-callback implementation, without editing
the upstream model or training code. Use eager inference, one active sampling
request per model instance; this adapter is not a concurrent-hook or
torch.compile integration.

G03 numerical components derive from the measured Reference7 G03 runtime.
The generic `DiFA.refine` core consumes grouped tensors, allows any positive
branch count and has no model/CFG/solver imports. `SpatialReference` provides
model-independent BCHW consensus and explicit raw history transactions.
`difa/adapters/sit.py` alone owns SiT coordinates, branch observations and
the fixed-Heun calling protocol. The two original incremental additions are
retained in order to preserve the measured G03 arithmetic.

Generic tests also cover non-image grouped tensors, 1/3/4 response branches,
arbitrary feature dimensions, rectangular spatial layouts and explicit history
read/commit boundaries. Tensor-interface generality does not establish quality
improvements on other models. See docs/DIFA_API.md for the coordinate contract.

Local CPU fixture validation: identical pretrained-state key structure,
unchanged ordinary SiT/CFG outputs, and bitwise equality of the entire G03
trajectory and diagnostics against the measured callback implementation.
The 20-node trajectory uses 38 full forwards and 18 same-time pairs, with no
extra random draws, weight updates or additional block forwards. This checks
the integration; the reported FID comes from the original developer-machine
50k run, not a new evaluation of this source layout.

SiT's license and attribution are preserved in LICENSE.txt and README_SiT.md.
The new DiFA inference additions are identified above; upstream copyright
attribution is not an assertion that upstream authors wrote these additions.
