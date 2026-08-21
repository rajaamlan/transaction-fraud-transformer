# Benchmarks

## Primary Model

- FT-Transformer trained on IEEE-CIS.
- Saved as `artifacts/model.pt`.
- Served by the FastAPI fraud agent.

## Benchmark Models

- LightGBM: strong classic tabular baseline.
- TabPFN: pretrained tabular foundation baseline.
- TabFM: Google zero-shot tabular foundation model; check license before any
  commercial use.
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
