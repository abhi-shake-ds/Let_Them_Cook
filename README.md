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

The repository will follow this structure:

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
│   └── README.md
│
├── results/
│   ├── metrics.csv
│   └── figures/
│
├── report/
│   └── project-report.pdf
│
├── requirements.txt
├── README.md
└── REPRODUCIBILITY.md
```

*Note: The structure describes the planned organization. Files and directories should be added as the project develops. The report, metrics, and figures will be generated from the completed analysis.*

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
- Encoding cyclical variables such as hour and month using sine and cosine transformations.
- Comparing cyclical encoding with raw hour values and one-hot hour encoding.
- Encoding categorical variables such as season, weather situation, and weekday.
- Using scikit-learn's `ColumnTransformer` and `Pipeline` for preprocessing.
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

Temperature and feeling-temperature variables will be inspected in their original normalized form. Any conversion to degrees Celsius will be documented.

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
| Random Forest or Gradient Boosting | Capture nonlinear relationships and feature interactions |

The initial Linear Regression model will use a small feature set, including hour, temperature, working-day status, and year.

Subsequent models will use engineered features and a consistent preprocessing pipeline.

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

**Important:** Actual model scores and conclusions will be added after training and evaluating the models. No performance results are assumed in advance.

## 📈 Expected Outcomes

The project aims to produce:

- An exploratory analysis of hourly bike-rental demand.
- A documented preprocessing and feature-engineering workflow.
- A baseline forecasting model.
- A comparative evaluation of linear, regularized, and tree-based models.
- A common metrics table for all evaluated models.
- Residual plots and interpretable findings.
- A final report describing the methodology, results, limitations, and potential applications to fleet rebalancing.

These are planned deliverables, not claims of completed results.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version.
- Git.
- Jupyter Notebook or JupyterLab.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Let_Them_Cook
```

Replace the placeholder with the actual repository URL.

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

Create `requirements.txt` containing:

```text
pandas
numpy
scikit-learn
matplotlib
jupyter
```

Install the packages:

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the UCI Bike Sharing Dataset from the official source:

[Download the dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

Extract the archive and place `hour.csv` and `day.csv` inside `data/raw/`.

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
- Building a persistence baseline using demand from the same hour in the previous week.
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
**Status:** In Development
