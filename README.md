# Linear Regression Taxi Fare Prediction

## Project Overview

This project builds a linear regression model to predict taxi fare prices in Chicago, Illinois.

The notebook uses a real taxi trip dataset, explores the relationship between trip distance, trip duration, and fare, then trains machine learning models to estimate taxi fares.

The project is based on a Google Machine Learning Crash Course Colab exercise.

## Objectives

- Load taxi trip data into a pandas DataFrame.
- Explore the dataset using descriptive statistics.
- Identify features related to taxi fare.
- Visualize relationships between fare, trip distance, and trip duration.
- Train a linear regression model using Keras.
- Tune model hyperparameters.
- Compare one-feature and two-feature models.
- Validate fare predictions against observed fare values.

## Dataset

The notebook uses the Chicago taxi training dataset from Google ML Education.

Dataset URL:

```text
https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv
```

The dataset is derived from the City of Chicago Taxi Trips dataset.

## Selected Features

The notebook selects these columns:

| Column | Description |
|---|---|
| TRIP_MILES | Distance of the taxi trip in miles |
| TRIP_SECONDS | Duration of the taxi trip in seconds |
| FARE | Actual taxi fare |
| COMPANY | Taxi company name |
| PAYMENT_TYPE | Payment method |
| TIP_RATE | Tip rate |
| TRIP_MINUTES | Derived feature from TRIP_SECONDS divided by 60 |

## Dataset Summary

Key dataset details from the notebook:

| Item | Value |
|---|---:|
| Total rows | 31,694 |
| Maximum fare | $159.25 |
| Mean trip distance | 8.2895 miles |
| Number of taxi companies | 31 |
| Most frequent payment type | Credit Card |
| Missing values | None reported |

## Correlation Findings

The correlation matrix shows:

| Feature | Correlation with FARE |
|---|---:|
| TRIP_MILES | 0.9753 |
| TRIP_SECONDS | 0.8303 |
| TIP_RATE | -0.0710 |

Main finding:

`TRIP_MILES` has the strongest relationship with `FARE`, so it is the best starting feature for the linear regression model.

## Methodology

### 1. Data Loading

The dataset is loaded using pandas:

```python
chicago_taxi_dataset = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
)
```

### 2. Data Selection

The notebook filters the dataset to keep only relevant columns:

```python
training_df = chicago_taxi_dataset.loc[:, (
    "TRIP_MILES",
    "TRIP_SECONDS",
    "FARE",
    "COMPANY",
    "PAYMENT_TYPE",
    "TIP_RATE"
)]
```

### 3. Exploratory Data Analysis

The notebook performs:

- Descriptive statistics using `DataFrame.describe()`.
- Correlation analysis using `DataFrame.corr()`.
- Pairwise visualization using Plotly scatter matrix.

### 4. Model Building

The model uses a simple Keras neural network structure equivalent to linear regression:

- Input layer based on selected feature columns.
- Concatenation layer for multiple features.
- Dense output layer with one unit.
- RMSprop optimizer.
- Mean squared error loss.
- Root mean squared error as the evaluation metric.

### 5. Experiments

The notebook runs three main experiments.

| Experiment | Features | Purpose |
|---|---|---|
| Experiment 1 | TRIP_MILES | Train a baseline one-feature model |
| Experiment 2 | TRIP_MILES | Test hyperparameter effects |
| Experiment 3 | TRIP_MILES, TRIP_MINUTES | Train a stronger two-feature model |

## Hyperparameters

Default training settings used in the notebook:

| Hyperparameter | Value |
|---|---:|
| Learning rate | 0.001 |
| Epochs | 20 |
| Batch size | 50 |
| Metric | RMSE |

## Model Insight

The two-feature model uses:

```text
TRIP_MILES
TRIP_MINUTES
```

This improves the model because taxi fare depends on both distance and time.

The notebook also compares the trained model with the known Chicago taxi fare logic:

```text
FARE = 2.25 * TRIP_MILES + 0.12 * TRIP_MINUTES + 3.25
```

A good model should learn weights close to this fare structure.

## Prediction Output

The notebook validates the model by producing a prediction table with:

| Column | Description |
|---|---|
| PREDICTED_FARE | Model prediction |
| OBSERVED_FARE | Actual fare from dataset |
| L1_LOSS | Absolute error between predicted and observed fare |
| TRIP_MILES | Trip distance |
| TRIP_MINUTES | Trip duration in minutes |

## Technologies Used

- Python
- pandas
- NumPy
- Keras
- TensorFlow
- Plotly
- Matplotlib
- Google ML Education package
- Jupyter Notebook or Google Colab

## Installation

### Recommended Option: Google Colab

Open the notebook in Google Colab and run the cells from top to bottom.

### Local Setup

Create a clean environment before running the notebook locally.

```bash
conda create -n taxi_regression python=3.11 -y
conda activate taxi_regression
```

Install dependencies:

```bash
pip install google-ml-edu==0.1.3
pip install keras tensorflow pandas numpy matplotlib plotly jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

## Troubleshooting

### Keras, TensorFlow, SciPy, or NumPy Import Error

A common error is:

```text
AttributeError: module 'numpy' has no attribute '_no_nep50_warning'
```

This usually means the local Anaconda environment has incompatible versions of NumPy, SciPy, TensorFlow, or Keras.

Recommended fix:

```bash
conda create -n taxi_ml python=3.11 -y
conda activate taxi_ml

python -m pip install numpy==1.26.4 scipy==1.11.4 pandas scikit-learn matplotlib plotly jupyter
python -m pip install tensorflow keras google-ml-edu

python -m ipykernel install --user --name taxi_ml --display-name "Python (taxi_ml)"
```

Then select the new kernel in Jupyter:

```text
Kernel > Change Kernel > Python (taxi_ml)
```

## How to Run

1. Open `linear_regression_taxi.ipynb`.
2. Run the setup and dependency cells.
3. Load the Chicago taxi dataset.
4. Explore descriptive statistics.
5. Review the correlation matrix.
6. Train the one-feature model.
7. Run the hyperparameter experiment.
8. Train the two-feature model.
9. Compare model performance.
10. Review fare predictions.

## Expected Results

After running the notebook, you should see:

- Dataset preview.
- Dataset statistics.
- Correlation matrix.
- Scatter matrix visualization.
- RMSE loss curve.
- Model prediction plots.
- Comparison between one-feature and two-feature models.
- Fare prediction table.

## Project Structure

```text
.
├── linear_regression_taxi.ipynb
└── README.md
```

## Key Takeaways

- Trip distance is the strongest predictor of taxi fare.
- Trip duration also improves fare prediction.
- Scaling feature values matters when using multiple input features.
- A simple linear regression model gives useful fare estimates.
- RMSE helps compare model performance across experiments.

## License

The notebook includes Google LLC copyright and Apache License 2.0 information.

Review the original notebook license before redistribution.
