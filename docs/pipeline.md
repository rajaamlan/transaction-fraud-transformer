# Pipeline

```text
1. Kaggle data mount
2. LightGBM benchmark
3. FT-Transformer sanity run
4. FT-Transformer improved run
5. Save best artifacts
6. Download artifacts locally
7. Serve through FastAPI
8. Push to GitHub
9. Deploy Docker API
```

The Transformer is the primary model. LightGBM, TabPFN, TabFM, and TabICL are
benchmarks used to understand what the Transformer must beat.

## Current Scorecard

```text
LightGBM sanity, 50k rows:
  ROC-AUC 0.8913
  AUPRC   0.6082

FT-Transformer sanity, 50k rows, 3 epochs:
  ROC-AUC 0.8416
  AUPRC   0.3636
```

## Next Training Target

Run the improved Transformer config on Kaggle:

```text
150k rows
8 epochs
d_model 96
3 Transformer layers
best checkpoint selected by validation AUPRC
```
