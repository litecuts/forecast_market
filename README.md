
# Time Series Forecasting for Market Data (ARIMA + Prophet)

This project forecasts financial time series using **ARIMA** (statsmodels) and **Prophet** (if installed).
It downloads real prices from **Yahoo Finance** (via `yfinance`) and falls back to **synthetic data** if download fails.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python forecast_market.py --ticker AAPL --start 2018-01-01 --end 2024-12-31 --horizon 90
```

## Outputs
- `metrics.json` — RMSE & MAPE for ARIMA and Prophet
- `forecast_holdout.csv` — actual vs predictions on the holdout
- `actual_vs_forecast.png` — train vs test vs predictions
- `series_with_best_forecast.png` — full series with best model overlay
