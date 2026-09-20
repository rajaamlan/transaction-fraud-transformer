# Benchmarks

## Primary Model

- FT-Transformer trained on IEEE-CIS.
- Saved as `artifacts/model.pt`.
- The notebook checkpoint is used for Kaggle predictions. API artifact integration
  is still required; the existing FastAPI scaffold has a different artifact contract.

## Benchmark Models

- LightGBM: strong classic tabular baseline.
- TabPFN: installed, but benchmark blocked on pretrained-model license/access.
- TabFM: unimplemented research comparison, with no measured score.
- TabICL: tabular in-context learning baseline.

Benchmarks should report the same metrics:

```text
AUPRC
ROC-AUC
F1
rows
features
inference time
```

Measured results and evaluation groups are in [kaggle_verified_run.md](kaggle_verified_run.md).
The small matched benchmark cannot be ranked directly against the larger main run.
F1 is not reported for this release because no operating threshold was selected.
