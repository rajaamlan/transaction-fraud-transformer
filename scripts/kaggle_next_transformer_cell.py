"""Copy this cell into Kaggle for the next improved Transformer run."""

import json
import os
import subprocess
import sys


cmd = [
    sys.executable,
    "-m",
    "fraud_model.train",
    "--ieee-cis-dir",
    "/kaggle/input/competitions/ieee-fraud-detection",
    "--label-column",
    "isFraud",
    "--artifact-dir",
    "/kaggle/working/artifacts",
    "--sample-rows",
    "150000",
    "--epochs",
    "8",
    "--batch-size",
    "512",
    "--d-model",
    "96",
    "--nhead",
    "4",
    "--num-layers",
    "3",
    "--dropout",
    "0.15",
    "--device",
    "cuda",
]

print("Running:", " ".join(cmd))
env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/src:" + env.get("PYTHONPATH", "")
result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
print(result.stdout)
metrics = json.loads(result.stdout)
print("Best epoch:", metrics["best_epoch"])
print("Best AUPRC:", metrics["average_precision"])
