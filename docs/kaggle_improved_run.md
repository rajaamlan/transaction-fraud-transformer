# Kaggle Improved Transformer Run

Use this in a Kaggle notebook after enabling GPU T4 and attaching the IEEE-CIS
competition input.

## Cell 1

```python
import os
DATA_DIR = "/kaggle/input/competitions/ieee-fraud-detection"
print(os.listdir(DATA_DIR))
```

## Cell 2

```python
import torch
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
```

## Cell 3

```python
import sys
sys.path.append("/kaggle/working/src")
```

If you uploaded the repo to Kaggle, update the path to where `src/` lives.

## Cell 4

```python
from fraud_model.train import parse_args, train

args = parse_args([
    "--ieee-cis-dir", "/kaggle/input/competitions/ieee-fraud-detection",
    "--label-column", "isFraud",
    "--artifact-dir", "/kaggle/working/artifacts",
    "--sample-rows", "150000",
    "--epochs", "8",
    "--batch-size", "512",
    "--d-model", "96",
    "--nhead", "4",
    "--num-layers", "3",
    "--dropout", "0.15",
    "--lr-factor", "0.5",
    "--lr-patience", "1",
    "--max-grad-norm", "1.0",
    "--early-stopping-patience", "2",
    "--device", "cuda",
])

metrics = train(args)
metrics
```

## Cell 5

```python
import json
from pathlib import Path

artifact_dir = Path("/kaggle/working/artifacts")
print(json.loads((artifact_dir / "metrics.json").read_text()))
print((artifact_dir / "model.pt").exists())
print((artifact_dir / "preprocessor.json").exists())
```

## What this run does

- uses IEEE-CIS as the training source
- samples 150k rows for a stronger but still manageable run
- trains the Transformer with early stopping and learning-rate scheduling
- saves the best checkpoint by validation AUPRC
- writes model artifacts for later download and GitHub deployment
