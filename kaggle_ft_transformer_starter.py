"""Kaggle starter for training the fraud FT-Transformer.

Paste this file into a Kaggle notebook cell with the IEEE-CIS input attached,
or upload the repository and run it from the notebook.
"""

import json
import os
import sys
from pathlib import Path


REPO_DIR = Path("/kaggle/working/transaction-fraud-transformer")
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR / "src"))
else:
    sys.path.insert(0, "/kaggle/working/src")

from fraud_model.train import parse_args, train  # noqa: E402


DATA_DIR = "/kaggle/input/competitions/ieee-fraud-detection"
ARTIFACT_DIR = "/kaggle/working/artifacts"


args = parse_args(
    [
        "--ieee-cis-dir",
        DATA_DIR,
        "--label-column",
        "isFraud",
        "--artifact-dir",
        ARTIFACT_DIR,
        "--sample-rows",
        "50000",
        "--epochs",
        "3",
        "--batch-size",
        "512",
        "--device",
        "cuda",
    ]
)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
metrics = train(args)
print(json.dumps(metrics, indent=2))
