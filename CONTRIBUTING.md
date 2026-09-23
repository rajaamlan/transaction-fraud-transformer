# Contributing

Thanks for checking out the project. The best contributions here are small,
reproducible, and honest about model limits.

## Good First Contributions

- Improve documentation or examples.
- Add tests for feature engineering edge cases.
- Add leakage checks for time-series features.
- Improve the stock forecasting walkthrough.
- Add benchmark scripts that are easy to rerun.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q
```

## Contribution Rules

- Keep raw datasets, credentials, large checkpoints, and generated submissions
  out of git.
- Fit preprocessors only on training data.
- Do not use future data in model features.
- Include tests for new reusable code.
- Report model results with enough context to reproduce them.

## Financial Forecasting Note

The stock forecaster is research-only and is not financial advice. Any future
broker or portfolio automation must go through backtesting, paper trading, risk
limits, and human approval first.

