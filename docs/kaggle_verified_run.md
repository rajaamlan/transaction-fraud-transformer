# Kaggle Verification - 2026-09-11

Notebook: https://www.kaggle.com/code/rajaamlan/notebook70df114be2/edit

Source: `notebooks/building-transformers-v2.ipynb` (outputs cleared for Git).

## Main Experiment

IEEE-CIS: 50,000 sampled rows, 40,000 training and 10,000 validation;
stratified random split, seed 42, 432 features. This is development validation,
not a chronological deployment evaluation or an untouched final test set.

| Model | ROC-AUC | Average precision (AUPRC) |
| --- | --- | --- |
| LightGBM | 0.891275 | 0.608177 |
| FT-Transformer, best epoch 5 | 0.853336 | 0.417455 |

Transformer: dimension 64, 4 attention heads, 2 layers, batch size 64.
Preprocessing is fitted on training rows only, including imputation and scaling.
Validation runs in batches; the full validation tensor is never passed to GPU attention.
The corrected run completed on a Tesla T4 without a CUDA out-of-memory error.
Saved preprocessing and checkpoint were reloaded on CPU and predictions matched
the validation predictions within rtol=1e-4 and atol=1e-5.

## Resource-Limited Foundation Comparison

Both successful models used the same 2,000 training rows, 1,000 validation rows,
and 100 features selected using training labels only. These scores are not
directly comparable with the larger main experiment above. Only about 36 positive
examples are in this validation subset, so treat results as preliminary.

| Model | ROC-AUC | Average precision (AUPRC) | Status |
| --- | --- | --- | --- |
| LightGBM matched | 0.575251 | 0.144277 | Passed |
| TabICL | 0.823262 | 0.254056 | Passed |
| TabPFN | - | - | Requires Prior Labs license acceptance and model-access token |
| TabFM | - | - | Not implemented |

Verified installs: `tabicl==2.2.0`, `tabpfn==8.5.0`.
TabICL used `tabicl-classifier-v2-20260212.ckpt`, one ensemble member and
ensemble batch size 1. Inference queries were batched in groups of 64.

## Remaining Work

Finalization on 2026-09-19 reused this verified checkpoint for all 506,691 test rows.
Kaggle version 2 completed as a late submission: public ROC-AUC **0.867624**,
private ROC-AUC **0.844891**. See [submission.md](submission.md) for provenance.

- User accepts the TabPFN license through https://ux.priorlabs.ai and configures
  the resulting token privately in Kaggle Secrets; never commit credentials.
- Use a chronological split and an untouched test period before deployment claims.
- Adapt this notebook checkpoint and preprocessing bundle to the repository API's
  artifact contract before deployment; matching filenames do not imply compatibility.
