# SiT ImageNet evaluation

The evaluation entry point uses the
[ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations)
to compute FID, sFID, Inception Score, Precision, and Recall from SiT samples.

## Generate comparison datasets

From the repository root, after preparing the pretrained assets:

```bash
python sample.py --no-difa --solver heun --num-steps 20 \
  --num-samples 50000 --batch-size 32 --seed 1 \
  --output-dir ./outputs/imagenet256_baseline --skip-packing

python sample.py --difa --solver heun --num-steps 20 \
  --num-samples 50000 --batch-size 32 --seed 1 \
  --output-dir ./outputs/imagenet256_difa --skip-packing
```

Use the same solver, step count, seed, sample count, batch size, and GPU count
for both modes. The supported solvers are `euler` and `heun`. Select a batch
size appropriate for the available GPU memory before generating both datasets.
Each output directory contains `run_config.json` with the selected settings.

## Pack samples on CPU

Pack the numbered RGB PNGs without running the model again:

```bash
python eval.py --sample-dir ./outputs/imagenet256_baseline --num-samples 50000 --pack-only
python eval.py --sample-dir ./outputs/imagenet256_difa --num-samples 50000 --pack-only
```

Each `<sample_dir>.npz` contains `arr_0` with shape `[50000,256,256,3]` and
dtype `uint8`. Packing reads one RGB image at a time. CPU packing of a
completed dataset can run concurrently with GPU sampling of the next dataset.

## Evaluate with ADM

Obtain the ImageNet 256 × 256 reference NPZ and Inception graph following the
ADM evaluation instructions. Store them at:

- `data/references/VIRTUAL_imagenet256_labeled.npz`
- `data/evaluation/classify_image_graph_def.pb`

Install the backend and evaluation dependencies, then evaluate each dataset:

```bash
python -m pip install -r requirements-eval.txt
python scripts/setup_evaluation.py

python eval.py --samples-npz ./outputs/imagenet256_baseline.npz \
  --reference-statistics ./data/references/VIRTUAL_imagenet256_labeled.npz \
  --inception-graph ./data/evaluation/classify_image_graph_def.pb \
  --output-file ./outputs/baseline_metrics.json

python eval.py --samples-npz ./outputs/imagenet256_difa.npz \
  --reference-statistics ./data/references/VIRTUAL_imagenet256_labeled.npz \
  --inception-graph ./data/evaluation/classify_image_graph_def.pb \
  --output-file ./outputs/difa_metrics.json
```

Keep the reference dataset, graph, preprocessing, and image count the same
across methods. The evaluator saves all five metrics and evaluation paths in
JSON. Evaluation uses CPU; TensorFlow is not required for sample generation.
