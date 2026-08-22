# Kaggle Workflow

Use IEEE-CIS as a competition input. Kaggle may mount it here:

```text
/kaggle/input/competitions/ieee-fraud-detection
```

Confirm GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Run sanity training:

```bash
PYTHONPATH=src python -m fraud_model.train \
  --ieee-cis-dir /kaggle/input/competitions/ieee-fraud-detection \
  --label-column isFraud \
  --artifact-dir /kaggle/working/artifacts \
  --sample-rows 50000 \
  --epochs 3 \
  --batch-size 512 \
  --device cuda
```

Run the next improved training:

```bash
PYTHONPATH=src python -m fraud_model.train \
  --ieee-cis-dir /kaggle/input/competitions/ieee-fraud-detection \
  --label-column isFraud \
  --artifact-dir /kaggle/working/artifacts \
  --sample-rows 150000 \
  --epochs 8 \
  --batch-size 512 \
  --d-model 96 \
  --nhead 4 \
  --num-layers 3 \
  --dropout 0.15 \
  --lr-factor 0.5 \
  --lr-patience 1 \
  --max-grad-norm 1.0 \
  --early-stopping-patience 2 \
  --device cuda
```

Download these artifacts after training:

```text
artifacts/model.pt
artifacts/preprocessor.json
artifacts/metrics.json
```

For a notebook-style step list, see [docs/kaggle_improved_run.md](docs/kaggle_improved_run.md).
