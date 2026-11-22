# 🚲 Let Them Cook — Bike Rental Demand Forecasting

**Project ID:** P01  
**Project Title:** Hourly Bike-Rental Demand for Fleet Rebalancing  
**Team Name:** Let Them Cook

## 📌 Project Overview

This project focuses on forecasting hourly bike-rental demand using the **UCI Bike Sharing Dataset** and machine learning techniques implemented with Python and scikit-learn.

Accurate demand forecasting can help bike-sharing operators make better fleet-rebalancing decisions by anticipating periods of high and low demand. This can improve bike availability, reduce shortages at busy locations, and support more efficient fleet utilization.

The project explores and compares linear and regularized regression models with tree-based models. It also investigates feature engineering, temporal patterns, data preprocessing, and evaluation strategies suitable for time-dependent data.

A **chronological train/test split** is used to evaluate how well the models predict demand in a later period, avoiding the misleading results that can arise from randomly splitting time-series observations.

## 👥 Team Members

| S. No. | Name | Student ID |
|---:|---|---|
| 1 | Abhinav Parth Nautiyal | 590024730 |
| 2 | Abhinav Chaudhary | 590025274 |
| 3 | Mohit Singh Rathour | 590025840 |
| 4 | Abhishek Singh | 590027542 |
| 5 | Vanshika Pal | 590028075 |
| 6 | Tanisha Roy | 590027282 |
| 7 | Tanishka Jain | 590026822 |
| 8 | Diya Shrimali | 590022529 |

## 🎯 Objectives

- Analyze hourly and seasonal patterns in bike-rental demand.
- Explore relationships between demand, time, weather, and environmental conditions.
- Identify missing hours and investigate their implications for hourly forecasting.
- Build a baseline using Linear Regression.
- Compare Linear Regression, Ridge Regression, Lasso Regression, and a tree-based model.
- Apply feature engineering techniques for cyclical and categorical variables.
- Prevent target leakage and use a chronological evaluation strategy.
- Compare model performance using RMSE, MAE, and R².
- Derive insights that could support bike-fleet planning and rebalancing.

## 📊 Dataset

**Dataset:** UCI Bike Sharing Dataset  
**Source:** [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

The dataset contains bike-rental records collected by the Capital Bikeshare system in Washington, D.C., during 2011 and 2012, along with weather and calendar-related attributes.

The project primarily uses `hour.csv`, which contains hourly rental records.

The original hourly dataset contains **17,379 rows and 17 columns**.

### Important Features

| Feature | Description |
|---|---|
| `dteday` | Date of the observation |
| `season` | Season identifier |
| `yr` | Year indicator |
| `mnth` | Month |
| `hr` | Hour of the day |
| `holiday` | Whether the day is a holiday |
| `weekday` | Day of the week |
| `workingday` | Whether the day is a working day |
| `weathersit` | Weather situation category |
| `temp` | Normalized temperature |
| `atemp` | Normalized feeling temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `casual` | Count of casual users |
| `registered` | Count of registered users |
| `cnt` | Total number of bike rentals |

The `cnt` column is the target variable.

**Target definition:**

`cnt = casual + registered`

Because `casual` and `registered` directly determine the target, they will be excluded from the model features to prevent target leakage. The date and record-index columns will also be excluded from the predictive feature matrix where appropriate.

## 🛠️ Technology Stack

- **Language:** Python
- **Data manipulation:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Visualization:** Matplotlib
- **Exploratory analysis:** Jupyter Notebook
- **Version control:** Git and GitHub

## 📁 Repository Structure

The repository has this structure:

```text
Let_Them_Cook/
│
├── data/
│   └── raw/
│       ├── hour.csv
│       └── day.csv
│
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-models.ipynb
│   └── 04-evaluation.ipynb
│
├── src/
│   ├── README.md
│   ├── data.py              loading, chronological split, X/y separation
│   ├── preprocessing.py     'linear' and 'forest' preprocessing pipelines
│   └── baselines.py         persistence (same hour last week) forecast
│
├── results/
│   ├── metrics.csv  best_params.csv
│   ├── preds_baseline.csv  preds_models.csv
│   └── figures/
│
├── report/
│   └── report.pdf           (to be added)
│
├── requirements.txt
├── README.md
└── REPRODUCIBILITY.md
```

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)

The first stage examines the dataset's structure, quality, and demand patterns.

Planned analyses include:

- Inspecting data types, dimensions, and missing values.
- Checking the continuity of the hourly timeline.
- Investigating missing timestamps and distinguishing absent records from confirmed zero-rental periods.
- Plotting average hourly demand for working and non-working days.
- Examining demand over time to identify trends and seasonal patterns.
- Studying relationships between rental demand and weather conditions.

**Expected visualizations:**

- Average hourly rentals by working-day status.
- Rental demand over the observation period.
- Demand patterns across seasons and weather conditions.

### 2. Data Preprocessing and Feature Engineering

Preprocessing will prepare the features for consistent and reliable model training.

Planned steps include:

- Removing `casual`, `registered`, `dteday`, and `instant` from the predictive feature matrix.
- Retaining `yr` as a predictor.
- Comparing raw, cyclical (sine/cosine) and one-hot hour encoding, and raw versus cyclical month, with `TimeSeriesSplit` on the training period.
- Testing a one-hot **hour x working-day** encoding for the linear models (lowest cross-validated RMSE, better in all 5 folds).
- Encoding categorical variables such as season, weather situation, and weekday.
- Using scikit-learn's `ColumnTransformer` and `Pipeline` for preprocessing. Two variants are used (`src/preprocessing.py`): `'linear'` (one-hot hour x working day, `atemp` dropped because it is almost identical to `temp`) for Linear/Ridge/Lasso, and `'forest'` (cyclical hour and month) for the random forest.
- Fitting transformations and scalers on training data only.

Cyclical encoding helps represent the fact that hour 23 is close to hour 0, and December is close to January.

For a cyclical feature \(x\) with period \(T\):

\[
x_{\sin}=\sin\left(\frac{2\pi x}{T}\right)
\]

\[
x_{\cos}=\cos\left(\frac{2\pi x}{T}\right)
\]

The periods used will be 24 for the hour and 12 for the month.

Temperature is normalized in the file. Degrees Celsius are obtained with `temp * 47 - 8` (minimum -8, maximum +39 according to the UCI documentation).

### 3. Chronological Train/Test Split

The dataset will be divided according to time rather than randomly.

| Dataset | Period | Purpose |
|---|---|---|
| Training set | January 2011 – June 2012 | Model fitting and training-period validation |
| Test set | July 2012 – December 2012 | Final model evaluation |

This split simulates a real forecasting scenario in which historical data is used to predict future demand.

The test set will be kept separate from model selection and hyperparameter tuning. `TimeSeriesSplit` and `GridSearchCV` will be used for tuning on the training period.

### 4. Model Development

The project will evaluate the following models:

| Model | Purpose |
|---|---|
| Linear Regression | Establish a baseline and identify linear relationships |
| Ridge Regression | Apply L2 regularization to reduce the impact of large coefficients |
| Lasso Regression | Apply L1 regularization and potentially reduce less useful coefficients to zero |
| Random Forest | Capture nonlinear relationships and feature interactions |

The initial Linear Regression model will use a small feature set, including hour, temperature, working-day status, and year.

Two reference forecasts are reported next to the models: this 4-feature baseline and a **persistence** forecast (demand of the same hour one week earlier, `src/baselines.py`). A model is only useful if it beats persistence.

Subsequent models use the engineered features described above.

Ridge and Lasso regularization strengths will be selected through cross-validation on the training period only.

### 5. Model Evaluation

Every model will be evaluated on the same held-out test period.

The following metrics will be reported:

**Root Mean Squared Error (RMSE)**

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

RMSE penalizes larger prediction errors more heavily and is expressed in the same units as bike rentals.

**Mean Absolute Error (MAE)**

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

MAE measures the average absolute prediction error and is also expressed in bikes.

**Coefficient of Determination (R²)**

\[
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat y_i)^2}{\sum_{i=1}^{n}(y_i-\bar y)^2}
\]

R² measures how much of the variation in the test-set target is explained by the model relative to predicting the test-set mean.

Results will be saved to `results/metrics.csv`.

Additional evaluation will include:

- Residuals against hour of the day.
- Residuals against predicted demand.
- Examination of errors during peak-demand periods.
- Interpretation of selected Ridge Regression coefficients.
- Analysis of the relationship between poor weather and rental demand.

RMSE is the planned headline metric because its units correspond directly to the number of bikes, making forecast errors easier to interpret for fleet planning.

## Results

All models are scored on the same test period (1 July 2012 to 31 December 2012, 4,376 hours).

| Model | RMSE (bikes/hour) | MAE (bikes/hour) | R² |
|---|---:|---:|---:|
| Baseline linear regression (4 features) | 191.3 | 145.3 | 0.242 |
| Persistence (same hour last week) | 107.9 | 63.1 | 0.759 |
| Linear regression | 107.1 | 80.2 | 0.762 |
| Ridge | 107.1 | 80.2 | 0.762 |
| Lasso | 107.1 | 80.2 | 0.762 |
| **Random forest** | **83.5** | **56.3** | **0.856** |

- The random forest is the best model and the only one that clearly beats the persistence forecast (RMSE about 23% lower). The linear models only match persistence in RMSE and are worse in MAE.
- With one-hot hour x working-day encoding the linear models improved from RMSE 165.7 to 107.1. Linear, Ridge and Lasso are almost identical because cross-validation chose the smallest penalty: with 13,003 training rows and 70 columns there is little to regularize.
- All models make their largest errors, and under-predict, in the busiest hours (the forest is about 136 bikes too low on average in the top 10% of hours).
- Bad weather lowers demand (about 64 fewer rentals per hour in light rain or snow than in clear weather), and 2012 had about 83 more rentals per hour than 2011.
- RMSE is the headline metric because it is in bikes.

## 📈 Deliverables

The project produces:

- An exploratory analysis of hourly bike-rental demand.
- A documented preprocessing and feature-engineering workflow.
- A baseline forecasting model.
- A comparative evaluation of linear, regularized, and tree-based models.
- A common metrics table for all evaluated models.
- Residual plots and interpretable findings.
- A final report describing the methodology, results, limitations, and potential applications to fleet rebalancing.

The final report (`report/report.pdf`) is still to be added.

## ⚙️ Installation and Setup

### Prerequisites

- The Python version listed in REPRODUCIBILITY.md (tested with Python 3.12.9).
- Git.
- Jupyter Notebook or JupyterLab.

### 1. Clone the repository

```bash
git clone https://github.com/abhi-shake-ds/Let_Them_Cook
cd Let_Them_Cook
```


### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages:

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

`hour.csv` and `day.csv` are already included in `data/raw/`, so there is nothing to download.

### 5. Run the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Execute the notebooks in the following order:

1. `01-eda.ipynb`
2. `02-preprocessing.ipynb`
3. `03-models.ipynb`
4. `04-evaluation.ipynb`

Run the notebooks from top to bottom to reproduce the analysis.

## 🔁 Reproducibility and Version Control

The project will use Git to maintain a record of development and important milestones.

Planned checkpoints:

- **Checkpoint 1 — Baseline:** Initial EDA, chronological split, and baseline Linear Regression evaluation.
- **Checkpoint 2 — Model Development:** Engineered features, regularized regression, and a tree-based model.
- **Final Submission:** Evaluation results, visualizations, report, README, and reproducibility documentation.

Checkpoint tags will be created as follows:

```bash
git tag checkpoint-1
git push origin checkpoint-1
```

```bash
git tag checkpoint-2
git push origin checkpoint-2
```

The `REPRODUCIBILITY.md` file will document the environment, dataset preparation, execution order, preprocessing decisions, random seeds where applicable, and steps required to regenerate the results.

## ⚠️ Methodological Considerations

The following practices are important to ensure a reliable comparison:

1. **Prevent target leakage:** Never include `casual` or `registered` as predictors of `cnt`.
2. **Respect temporal order:** Do not randomly split the observations into training and test sets.
3. **Avoid preprocessing leakage:** Fit scalers, encoders, and other learned transformations using training data only.
4. **Tune on training data:** Use time-aware cross-validation for hyperparameter selection.
5. **Keep evaluation consistent:** Compare all models on the same test period.
6. **Investigate missing timestamps:** An absent hourly record does not automatically prove that demand was zero. Verify the data and document any assumptions before filling gaps.
7. **Document limitations:** The dataset does not, by itself, provide every operational detail needed for station-level fleet rebalancing.

## 🚀 Future Enhancements

Potential extensions include:

- Implementing gradient descent manually to strengthen understanding of optimization.
- Classifying peak-demand versus non-peak-demand hours.
- Comparing additional feature-engineering approaches.
- Exploring prediction intervals or uncertainty estimates.
- Extending the analysis toward station-level demand forecasting if suitable data becomes available.

## 📚 References

1. Fanaee-T, H., & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence.
2. UCI Machine Learning Repository. [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
3. scikit-learn documentation:
   - [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
   - [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
   - [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
   - [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
   - [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)

---

**Project:** P01 — Hourly Bike-Rental Demand for Fleet Rebalancing  
**Team:** Let Them Cook  
**Domain:** Machine Learning · Regression · Time-Series Forecasting  
**Status:** Modelling and evaluation complete; final report to be added
