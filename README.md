# transaction-fraud-transformer
End-to-end fraud detection system for transaction data using a Transformer model, Kaggle GPU training, benchmark baselines, and a deployable FastAPI inference API.

# Transaction Fraud Transformer Agent

Licensed under the Apache License, Version 2.0.

This project adds a deployable fraud-scoring agent around a compact Transformer
architecture for tabular transaction data.

## Pipeline

1. Train the Transformer on IEEE-CIS using Kaggle GPU.
2. Save `model.pt`, `preprocessor.json`, and `metrics.json`.
3. Download artifacts into `artifacts/`.
4. Serve the trained Transformer through FastAPI.
5. Push the repo to GitHub.
6. Deploy with Docker.

LightGBM is only a benchmark. The primary model in this repo is the
Transformer.

See [docs/pipeline.md](docs/pipeline.md), [docs/kaggle.md](docs/kaggle.md),
and [docs/benchmarks.md](docs/benchmarks.md) for the working plan.

## Expected Columns

The default schema works with these fields:

- `amount`
- `hour`
- `merchant_risk`
- `customer_age_days`
- `transactions_last_24h`
- `country`
- `merchant_category`
- `payment_method`
- `is_fraud`

You can train with a different label name by passing `--label-column`.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python -m fraud_model.train `
  --data data/sample_transactions.csv `
  --label-column is_fraud `
  --artifact-dir artifacts
```

## Train On IEEE-CIS

On Kaggle, point `--ieee-cis-dir` at the competition folder:

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

Use the sample run first. Then remove `--sample-rows` and increase epochs for
the full training run.

The trainer saves the best checkpoint by validation AUPRC, not just the final
epoch.

Training writes:

- `artifacts/preprocessor.json`
- `artifacts/model.pt`
- `artifacts/metrics.json`

## Serve

```powershell
uvicorn fraud_model.api:app --host 0.0.0.0 --port 8000
```

Score one transaction:

```powershell
curl -X POST http://localhost:8000/score `
  -H "Content-Type: application/json" `
  -d "{\"amount\":349.99,\"hour\":2,\"merchant_risk\":0.8,\"customer_age_days\":12,\"transactions_last_24h\":9,\"country\":\"US\",\"merchant_category\":\"electronics\",\"payment_method\":\"card\"}"
```

## Docker

```powershell
docker build -t fraud-transformer-agent .
docker run -p 8000:8000 fraud-transformer-agent
```

## Deployment Options

The included Dockerfile can deploy to Render, Railway, Fly.io, AWS ECS, Azure
Container Apps, or Google Cloud Run. The GitHub Actions workflow runs tests and
builds the image on every push.

For a production deployment, connect persistent model artifacts through object
storage or bake a trained `artifacts/` directory into the image.
