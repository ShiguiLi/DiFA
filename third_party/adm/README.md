# ADM evaluation backend

The official evaluator and its license are downloaded separately from
[openai/guided-diffusion](https://github.com/openai/guided-diffusion).
From the DiFA repository root:

```bash
python scripts/setup_evaluation.py
```

This creates `third_party/adm/evaluator.py` and `third_party/adm/LICENSE`.
For an existing local guided-diffusion checkout, use:

```bash
python scripts/setup_evaluation.py --source-dir ./guided-diffusion
```

Obtain the compatible ImageNet reference NPZ and Inception graph using the
[upstream evaluation instructions](https://github.com/openai/guided-diffusion/tree/main/evaluations).
The root `eval.py` invokes this backend on CPU from the graph directory.
The graph must be named `classify_image_graph_def.pb`.
