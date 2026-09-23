# Transaction Fraud Transformer + Stock Forecasting Lab

[![CI](https://github.com/rajaamlan/transaction-fraud-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/rajaamlan/transaction-fraud-transformer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-submitted-20BEFF.svg)](https://www.kaggle.com/code/rajaamlan/notebook70df114be2)

A hands-on machine-learning repo for serious tabular modeling: an accepted
IEEE-CIS fraud-detection submission, reproducible transformer experiments,
matched tabular benchmarks, and a new stock price-range forecasting lab.

The north star is practical ML engineering: leakage-safe preprocessing,
measurable baselines, reproducible notebooks, tests, and API-ready scaffolding.

## Highlights

- **Accepted Kaggle late submission:** FT-Transformer predictions for all
  506,691 IEEE-CIS test transactions.
- **Verified leaderboard scores:** public ROC-AUC `0.867624`, private ROC-AUC
  `0.844891`.
- **Transformer-first tabular experiment:** FT-Transformer trained from scratch,
  with LightGBM and TabICL comparisons.
- **Production-minded scaffolding:** reusable scripts, tests, Dockerfile,
  FastAPI starter, and artifact validation.
- **New research direction:** stock price-range forecasting with uncertainty,
  backtesting, and eventual portfolio-assistant design.

## Project Tracks

| Track | Status | What It Shows |
| --- | --- | --- |
| Fraud Transformer | Verified | End-to-end tabular transformer experiment on IEEE-CIS |
| Kaggle Submission | Accepted late submission | Batched inference, CSV validation, reproducible artifacts |
| Tabular Benchmarks | In progress | LightGBM, TabICL, and future foundation-model comparisons |
| Stock Forecaster | Early lab | Leakage-safe financial features and baseline price ranges |

## Results Snapshot

### Kaggle Late Submission

| Model | Public ROC-AUC | Private ROC-AUC | Rows Scored |
| --- | ---: | ---: | ---: |
| FT-Transformer | **0.867624** | **0.844891** | 506,691 |

The submission was evaluated after the competition deadline, so it is not
prize-eligible. The machine-readable scorecard is stored in
[`results/scorecard.json`](results/scorecard.json).

### Main Validation Experiment

50,000 sampled transactions: 40,000 training rows, 10,000 validation rows,
432 features, seed 42. Preprocessing is fit on training rows only.

| Model | ROC-AUC | Average Precision |
| --- | ---: | ---: |
| LightGBM baseline | **0.891275** | **0.608177** |
| FT-Transformer | 0.853336 | 0.417455 |

This is intentionally reported honestly: the tree model wins the main validation
split, while the transformer path remains useful for learning architecture,
GPU-safe batching, and artifact handling.

## Stock Forecasting Lab

The stock track is a new build-from-scratch project for forecasting price ranges,
not exact future prices. The first version creates tabular features from OHLCV
data and fits a simple baseline range model before any XGBoost, LightGBM, or
transformer work.

Current stock files:

| Path | Purpose |
| --- | --- |
| [`docs/stock_forecaster_strategy.md`](docs/stock_forecaster_strategy.md) | Product and modeling strategy |
| [`src/stock_forecaster/features.py`](src/stock_forecaster/features.py) | Leakage-safe OHLCV feature engineering |
| [`src/stock_forecaster/baseline.py`](src/stock_forecaster/baseline.py) | Baseline expected price and range forecaster |
| [`tests/test_stock_forecaster.py`](tests/test_stock_forecaster.py) | Unit tests for feature and baseline behavior |

Planned output:

```text
Ticker: AAPL
Horizon: 5 trading days
Expected close: 234.10
Predicted range: 226.80 to 241.40
Historical error: +/- 3.2%
Signal: research-only, not financial advice
```

## Repository Layout

```text
.
+-- docs/                  # Experiment reports, roadmap, and strategy notes
+-- notebooks/             # Reproducible Kaggle notebook
+-- results/               # Machine-readable scorecards and run outputs
+-- scripts/               # Kaggle submission and notebook sync utilities
+-- src/
|   +-- fraud_model/       # Fraud detection training/API scaffold
|   +-- stock_forecaster/  # Stock price-range forecasting lab
+-- tests/                 # Unit tests and submission validation tests
+-- Dockerfile
+-- requirements.txt
+-- README.md
```

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q
```

If `pytest` is not installed yet, the stock-forecasting tests can also run with
the standard library test runner:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_stock_forecaster -v
```

## Reproduce The Kaggle Submission

1. Import [`notebooks/building-transformers-v2.ipynb`](notebooks/building-transformers-v2.ipynb)
   into Kaggle.
2. Attach the IEEE-CIS Fraud Detection dataset.
3. Select a T4 GPU.
4. Run training, validation, artifact checks, and final inference.
5. Submit the generated `submission.csv`.

More detail:

- [Verified experiment report](docs/kaggle_verified_run.md)
- [Submission guide](docs/submission.md)
- [Pipeline notes](docs/pipeline.md)

## Roadmap

Near-term work is tracked in [`docs/roadmap.md`](docs/roadmap.md). The main
themes are:

- add real stock data collection
- create a Jupyter walkthrough notebook for the stock lab
- add chronological validation for financial forecasting
- compare baseline, Random Forest, XGBoost, and LightGBM
- build a small dashboard once backtesting is honest

## Contributing

Contributions are welcome, especially around tests, leakage checks,
documentation, benchmarking, and small reproducible experiments. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Important Notes

- The fraud model is a learning and research project, not a deployed fraud
  prevention system.
- The stock forecaster is research-only and is not financial advice.
- Raw competition data, credentials, model checkpoints, and large binary
  artifacts are not committed.

## License

Code is released under the [Apache 2.0 License](LICENSE). Dataset terms,
competition rules, pretrained weights, and third-party services retain their own
licenses and restrictions.
