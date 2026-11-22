# Reproducibility Guide

## Environment

- Python: 3.12.9
- Packages: exact versions are in requirements.txt

## Setup (Windows)

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

On Linux or macOS use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Data

hour.csv and day.csv (UCI Bike Sharing) are already in data/raw/. The project uses hour.csv (17,379 rows, 17 columns).

## How to run

Open the notebooks in the notebooks folder and run Restart and Run All, in this order: 01-eda, 02-preprocessing, 03-models, 04-evaluation. Notebook 03 is the slowest because of the random forest search (a few minutes). Notebook 04 reads the files saved by notebook 03.

## Key decisions

- Chronological split: train before 2012-07-01 (13,003 rows), test from 2012-07-01 (4,376 rows).
- casual, registered, instant and dteday are never features (casual + registered = cnt, so using them is target leakage).
- The 165 missing hours (0.94%) are kept as missing, not filled.
- Final pipelines (`src/preprocessing.py`, everything fitted on training data only inside a Pipeline): `'linear'` for Linear, Ridge and Lasso (one-hot hour x working day, cyclic month, one-hot season, weathersit and weekday, StandardScaler for temp, hum and windspeed; atemp dropped because it is almost identical to temp). `'forest'` for the random forest (cyclic sin/cos for hour and month, one-hot for season, weathersit and weekday, StandardScaler for temp, atemp, hum and windspeed).
- A persistence forecast (same hour last week, `src/baselines.py`) is reported next to the models.
- Tuning: GridSearchCV with TimeSeriesSplit(5) on the training period only.
- Random seed: random_state=42.

## Files created

- 01-eda: figures in results/figures/
- 02-preprocessing: hour_encoding_comparison.png and month_encoding_comparison.png
- 03-models: results/preds_baseline.csv (4-feature baseline and persistence), results/preds_models.csv, results/best_params.csv
- 04-evaluation: results/metrics.csv, residual_vs_hour.png, residual_vs_predicted.png, ridge_coefficients.png, ridge_hour_profile.png

## Checkpoints

Git tags checkpoint-1 (baseline) and checkpoint-2 (models).