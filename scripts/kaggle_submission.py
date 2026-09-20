"""Notebook companion: chunked IEEE-CIS inference and strict CSV validation.

Run the function with the notebook's restored FTTransformer and preprocessing.pkl.
The notebook embeds this file so it does not need a GitHub checkout on Kaggle.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def validate_submission(submission, sample):
    if list(submission.columns) != ["TransactionID", "isFraud"]:
        raise ValueError("Submission must contain TransactionID,isFraud in that order")
    if submission.empty or len(submission) != len(sample):
        raise ValueError("Submission row count differs from sample_submission.csv")
    if not submission["TransactionID"].is_unique:
        raise ValueError("Duplicate transaction IDs")
    if not np.array_equal(submission["TransactionID"].to_numpy(), sample["TransactionID"].to_numpy()):
        raise ValueError("Transaction IDs or their order differ from the sample")
    probabilities = submission["isFraud"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Fraud predictions must be finite probabilities between zero and one")


def prepare_test_chunk(chunk, identity, preprocessing):
    # Test identity columns use id-XX; training used id_XX.
    identity = identity.rename(columns=lambda name: name.replace("id-", "id_"))
    merged = chunk.merge(identity, on="TransactionID", how="left", sort=False, validate="one_to_one")
    if not np.array_equal(merged["TransactionID"], chunk["TransactionID"]):
        raise ValueError("Identity join changed transaction ordering")
    numeric = preprocessing["numeric_features"]
    categorical = preprocessing["categorical_features"]
    missing = set(numeric + categorical) - set(merged.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    xn = preprocessing["scaler"].transform(
        merged[numeric].replace([np.inf, -np.inf], np.nan).fillna(preprocessing["numeric_medians"])
    ).astype("float32")
    xc = preprocessing["encoder"].transform(
        merged[categorical].fillna("__missing__").astype(str)
    ).astype("int64") + 1
    if not np.isfinite(xn).all() or (xc < 0).any():
        raise ValueError("Preprocessing produced invalid numeric values or category indices")
    return xn, xc


def generate_submission(model, preprocessing, data_dir, output_dir, device="cuda", batch_size=64):
    import torch

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    identity = pd.read_csv(data_dir / "test_identity.csv")
    if not identity["TransactionID"].is_unique:
        raise ValueError("Duplicate test identity IDs")
    model = model.to(device).eval()
    all_ids, all_predictions = [], []
    # Fix object dtypes across chunks; do not refit preprocessing on test data.
    category_types = {name: "object" for name in preprocessing["categorical_features"]}
    for chunk in pd.read_csv(data_dir / "test_transaction.csv", chunksize=10000, dtype=category_types):
        xn, xc = prepare_test_chunk(chunk, identity, preprocessing)
        predictions = []
        with torch.inference_mode():
            for offset in range(0, len(chunk), batch_size):
                logits = model(torch.from_numpy(xn[offset:offset+batch_size]).to(device),
                               torch.from_numpy(xc[offset:offset+batch_size]).to(device))
                predictions.append(torch.sigmoid(logits).cpu().numpy())
        all_ids.append(chunk["TransactionID"].to_numpy())
        all_predictions.append(np.concatenate(predictions))
        print(f"Predicted {sum(len(ids) for ids in all_ids):,} / {len(sample):,} rows", flush=True)
    predictions = pd.Series(np.concatenate(all_predictions), index=np.concatenate(all_ids))
    if not predictions.index.is_unique or set(predictions.index) != set(sample["TransactionID"]):
        raise ValueError("Test predictions do not cover exactly the sample transaction IDs")
    submission = sample[["TransactionID"]].copy()
    submission["isFraud"] = submission["TransactionID"].map(predictions)
    validate_submission(submission, sample)
    destination = output_dir / "submission.csv"
    submission.to_csv(destination, index=False)
    validate_submission(pd.read_csv(destination), sample)
    manifest = {
        "model": "ft_transformer_t4_safe", "rows": len(submission),
        "columns": list(submission.columns), "batch_size": batch_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "probability_min": float(submission.isFraud.min()),
        "probability_max": float(submission.isFraud.max()),
        "status": "validated_csv_not_yet_submitted", "training_rows": 40000,
    }
    (output_dir / "submission_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return submission
