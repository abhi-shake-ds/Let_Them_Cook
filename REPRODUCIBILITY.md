# Reproducibility Guide

## Environment

Python Version:
Python 3.14.6

## Installation

Create virtual environment:

python -m venv .venv

Activate environment:

Windows:
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Notebook Execution Order

1. notebooks/01-eda.ipynb
2. notebooks/02-preprocessing.ipynb
3. notebooks/03-models.ipynb
4. notebooks/04-evaluation.ipynb

## Project Settings

Split Date:
2012-07-01

Random State:
42

## Expected Outputs

01-eda.ipynb
- EDA figures in results/figures/

02-preprocessing.ipynb
- Encoding comparison figures

03-models.ipynb
- results/preds_baseline.csv
- results/preds_models.csv
- results/best_params.csv

04-evaluation.ipynb
- results/metrics.csv
- evaluation figures

Last updated by Abhishek Singh