# Kaggle Submission

IEEE-CIS exposes a **Late Submission** flow after its original competition.
The CSV columns are `TransactionID,isFraud`; each test transaction needs a
probability. The scoring metric is ROC-AUC, not average precision.

## Completed Submission

- Status: **Complete (after deadline)**, verified on Kaggle on 2026-09-19.
- Notebook version: 2, script version ID `351207506`.
- Public ROC-AUC: **0.867624**.
- Private ROC-AUC: **0.844891**.
- Test rows: **506,691**.
- CSV SHA-256: `50de370f267b1e5888a758c974de33ddd3d6b1d7c1bf486996456e85fe43a4b7`.

[Submitted version](https://www.kaggle.com/code/rajaamlan/notebook70df114be2?scriptVersionId=351207506).
The output manifest describes file generation; the final scored state is recorded
in [results/scorecard.json](../results/scorecard.json) after Kaggle evaluation.

## Reuse The Verified Checkpoint

Attach saved version 1 output of `rajaamlan/notebook70df114be2` using Add Input.
Run these cells in order:

1. Core imports.
2. Discover Kaggle dataset path.
3. GPU and memory check.
4. Define FT-Transformer.
5. Submission helper functions (section 7).
6. Restore the verified Transformer and generate the Kaggle submission.

For a fresh notebook, run training first if the original private output is not
accessible. Do not claim a rerun has identical scores unless measured.

## Validation

- Rename test identity columns `id-XX` to the training convention `id_XX`.
- Join identity records one-to-one, preserving transaction order.
- Reuse training medians, category mappings, and scaler without fitting test data.
- Read 10,000-row chunks and infer in batches of 64.
- Reject missing features, duplicate/missing IDs, nonfinite or out-of-range scores.
- Align to sample-submission order and revalidate the written CSV.
- Record checkpoint and CSV SHA-256 hashes in `submission_manifest.json`.

Save with **Save output for this version** or Save & Run All. Select
`submission.csv` in the submission dialog. Record a score only once Kaggle
reports completion.

## Deliverables

- `submission.csv`: Transformer predictions for every test transaction.
- `submission_manifest.json`: hashes, row count, checkpoint metrics.
- `results.json`: main validation and small matched benchmark in separate groups.
- `artifacts/`: checkpoint, fitted preprocessing, and verification outputs.

Generated data and weights belong in Kaggle outputs or an artifact store, not Git.
