# Filippo-Sallustio-s-Thesis-Repository
# FTSE MIB Forecasting

This repository demonstrates how to forecast the FTSE MIB index using
an LSTM-based neural network.
The data is provided in `dataftsemib_manual.csv`.

## Prerequisites

Install the required Python packages:

```bash
pip install pandas matplotlib scikit-learn tensorflow==2.12.0 statsmodels pmdarima
```

## Dataset summary

Basic descriptive statistics for `dataftsemib_manual.csv`:

|       |    Price |     Open |     High |      Low |           Vol. |   Change % |
|:------|---------:|---------:|---------:|---------:|---------------:|-----------:|
| count |  2810    |  2810    |  2810    |  2810    | 2793           |    2810    |
| mean  | 21626.3  | 21633    | 21788.8  | 21458.9  | 5.80479e+08    |       0.03 |
| std   |  3311.02 |  3307.02 |  3300.54 |  3313.23 | 2.7726e+08     |       1.42 |
| min   | 14894.4  | 14985.8  | 15267.8  | 14153.1  | 2.67e+06       |     -16.92 |
| 25%   | 19255.6  | 19266.4  | 19405.1  | 19090.9  | 3.8366e+08    |      -0.65 |
| 50%   | 21556.4  | 21544.2  | 21700.7  | 21395.5  | 4.9929e+08    |       0.08 |
| 75%   | 23607.7  | 23613.4  | 23759.6  | 23435.6  | 7.142e+08     |       0.78 |
| max   | 30426.6  | 30595    | 30652.9  | 30341.2  | 2.68e+09      |       8.93 |

## Running the LSTM example

Execute the LSTM script to clean the data, train the model and produce a
one-step ahead forecast.  By default the script expects
`dataftsemib_manual.csv`, but you can specify a different file with
`--data`.  You can also adjust the sequence window, batch size and
learning rate:

```bash
python3 ftse_mib_LSTM.py --epochs 100 --window 60 --batch 32
python3 ftse_mib_LSTM.py --data mydata.csv --lr 0.0005
```

If running inside Jupyter, ipykernel will pass additional command line
arguments.  The script ignores these using `parse_known_args`, so it can be
invoked with e.g. `!python ftse_mib_LSTM.py` inside a notebook without
raising `SystemExit` errors.

The script now uses several features (open, high, low, volume and a
5-day moving average) and feeds sequences of the last `--window` days
into a deeper LSTM network with dropout and learning rate scheduling.
Training stops automatically when the validation loss fails to improve
for several epochs.  Two plots
are saved and displayed:

- `closing_price_plot.png` – the cleaned closing prices
- `lstm_prediction_plot.png` – actual vs. predicted test prices

A trained model file `ftse_mib_lstm_model.h5` is saved along with the
best weights during training (`best_lstm.h5`).

## Feedforward ANN example

For a simpler neural approach without recurrent layers you can run the
ANN script.  It uses only the past 60 closing prices as inputs to a
multi-layer perceptron and forecasts the next day's close.

```bash
python3 ftse_mib_ann.py
```

The script scales the closing prices using `MinMaxScaler`, splits the
series into an 80/20 train/test split and trains a small network of
Dense layers with early stopping.  It prints RMSE, MAE, MAPE and R² on
the test portion and reports a residual Ljung–Box p-value to check for
autocorrelation.  Two figures are saved: `ann_prediction_plot.png`
showing actual versus predicted prices and `ann_residuals.png` with the
residual distribution.  The trained model is saved to
`ftse_mib_ann_model.h5`.

## ARIMA example

For a simple statistical baseline you can also run:

```bash
python3 ftse_mib_arima.py
```

This script displays and saves several diagnostic plots including the cleaned closing
prices (`arima_cleaned_prices.png`) as well as ACF/PACF graphs
(`arima_acf_pacf.png`). It then fits an ARIMA model selected via a small grid
search and produces a rolling one-step forecast to compare against the test
data. The series is split 80/20 between training and test portions. The results
are saved to `arima_prediction_plot.png`. If the residuals of the initial model
exhibit autocorrelation (Ljung-Box p < 0.05), the script automatically
re-estimates an ARIMA(3,1,3) model for comparison.

## XGBoost example

For a gradient boosting approach you can run an XGBoost model that forecasts
the next day's *log variance* using a large set of lagged return and volatility
features:

```bash
python3 ftse_mib_xgboost.py
```

The script constructs rolling statistics, range-based measures and technical
indicators, then tunes hyperparameters with a randomised search. Accuracy is
reported with RMSE, MAE, MAPE, $R^2$, QLIKE, directional accuracy and a
Ljung‑Box test on the residuals. The ten most important features are listed.
The high‑resolution plot `xgb_variance_prediction.png` compares the predicted
and actual log variance on the test set.


## EGARCH example

To model the volatility of daily returns you can run the EGARCH
script:

```bash
python3 ftse_mib_egarch.py
```

The script calculates log returns and incorporates log volume together
with lagged realised variance and past shocks as exogenous regressors.
After checking stationarity and ARCH effects it searches a few EGARCH
orders using Student's *t* and GED distributions.  The model forecasts
the next day's **log variance** (log of the squared return).  An 80/20
chronological split is evaluated and metrics including RMSE, MAE, MAPE,
$R^2$, QLIKE and directional accuracy are printed along with residual
statistics. Forecasts are saved as `egarch_variance_plot.png` together
with the model's AIC, BIC, log-likelihood and a Ljung-Box diagnostic.

## GJR-GARCH example

The threshold GARCH script cleans the dataset and computes log
returns.  Log volume, lagged realised variance and past shocks serve as
exogenous regressors.  A small grid search over `(p,o,q)` orders and
Student's *t* or GED distributions selects the model with the lowest
AIC.  Forecasts target the next day's **log variance** of returns.  An
80/20 split is used for evaluation with the same metrics as EGARCH and
residual diagnostics.

Run

```bash
python3 ftse_mib_gjrgarch.py
```

The script saves a high-resolution plot `gjrgarch_variance_plot.png` showing
actual versus predicted log variance and prints the same residual statistics
and information criteria as the EGARCH example.

## Model comparison

`compare_models.py` trains several forecasting approaches and applies the
Diebold‑Mariano test to compare their predictive accuracy.  Four model pairs
are examined:

- ARIMA vs ANN
- ARIMA vs LSTM
- EGARCH vs XGBoost
- GJR‑GARCH vs XGBoost

Run the comparison with

```bash
python3 compare_models.py
```

For GARC
