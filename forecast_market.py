
"""
forecast_market.py
Time Series Forecasting for Market Data using ARIMA (statsmodels) and Prophet (optional).

- Fetches real prices from Yahoo Finance (via yfinance) or generates synthetic data if download fails.
- Trains ARIMA and Prophet (if installed), evaluates on a holdout period, and saves plots + metrics.
- Outputs CSV of forecasts and a comparison JSON of error metrics (RMSE, MAPE).

Usage:
  python forecast_market.py --ticker AAPL --start 2018-01-01 --end 2024-12-31 --horizon 90
"""

import os, json, argparse, math, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def try_import_yf():
    try:
        import yfinance as yf
        return yf
    except Exception:
        return None

def try_import_prophet():
    # Try "prophet" (new name) then "fbprophet" (old)
    try:
        from prophet import Prophet
        return Prophet
    except Exception:
        try:
            from fbprophet import Prophet  # older versions
            return Prophet
        except Exception:
            return None

def download_prices_yf(ticker, start, end):
    yf = try_import_yf()
    if yf is None:
        return None
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()[["Date","Close"]].rename(columns={"Date":"date","Close":"close"})
        return df
    except Exception:
        return None

def generate_synth(start="2018-01-01", days=1500, s0=100.0, mu=0.08, sigma=0.25, seed=11):
    rng=np.random.default_rng(seed); dtv=1/252
    shocks=rng.normal((mu-0.5*sigma**2)*dtv, sigma*(dtv**0.5), size=days)
    prices=[s0]
    for e in shocks: prices.append(prices[-1]*math.exp(e))
    dates=pd.bdate_range(start=start, periods=days+1)
    return pd.DataFrame({"date":dates, "close":prices})

def train_test_split(df, test_horizon=90):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    train = df.iloc[:-test_horizon].copy()
    test  = df.iloc[-test_horizon:].copy()
    return train, test

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)

def model_arima(train, test, order=(5,1,0)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # Fit on train, forecast len(test)
    mod = SARIMAX(train["close"], order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    preds = res.forecast(steps=len(test))
    return preds, res

def model_prophet(train, test):
    Prophet = try_import_prophet()
    if Prophet is None:
        return None, None
    dfp = train[["date","close"]].rename(columns={"date":"ds","close":"y"})
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(dfp)
    future = pd.DataFrame({"ds": test["date"]})
    forecast = m.predict(future)
    preds = forecast["yhat"].values
    return preds, m

def plot_actual_vs_pred(train, test, pred_dict, out_path):
    plt.figure(figsize=(10,6))
    plt.plot(train["date"], train["close"], label="Train")
    plt.plot(test["date"], test["close"], label="Test (Actual)")
    for name, preds in pred_dict.items():
        if preds is None: continue
        plt.plot(test["date"], preds, label=f"Pred {name}")
    plt.title("Actual vs Forecast (Holdout)")
    plt.xlabel("Date"); plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_full_series_with_forecast(df, test, pred_best, label, out_path):
    plt.figure(figsize=(10,6))
    plt.plot(df["date"], df["close"], label="Actual")
    if pred_best is not None:
        plt.plot(test["date"], pred_best, label=f"Forecast ({label})")
    plt.title("Full Series with Forecast Overlay")
    plt.xlabel("Date"); plt.ylabel("Close")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def main():
    p=argparse.ArgumentParser(description="Time Series Forecasting with ARIMA and Prophet (optional)")
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--horizon", type=int, default=90, help="Holdout days for evaluation")
    p.add_argument("--arima_order", type=str, default="5,1,0", help="ARIMA order p,d,q")
    p.add_argument("--out_dir", type=str, default="outputs")
    args=p.parse_args()

    end=args.end or dt.date.today().isoformat()
    df = download_prices_yf(args.ticker, args.start, end)
    if df is None or df.empty:
        print("⚠️ Yahoo Finance download failed. Using synthetic data instead.")
        df = generate_synth(start=args.start)
    # split
    train, test = train_test_split(df, test_horizon=args.horizon)

    # ARIMA
    p_,d_,q_ = [int(x) for x in args.arima_order.split(",")]
    arima_preds, arima_model = model_arima(train, test, order=(p_,d_,q_))

    # Prophet (optional)
    prophet_preds, prophet_model = model_prophet(train, test)

    # Metrics
    metrics = {}
    metrics["ARIMA"] = {
        "RMSE": rmse(test["close"], arima_preds),
        "MAPE_pct": mape(test["close"], arima_preds),
    }
    if prophet_preds is not None:
        metrics["Prophet"] = {
            "RMSE": rmse(test["close"], prophet_preds),
            "MAPE_pct": mape(test["close"], prophet_preds),
        }
    else:
        metrics["Prophet"] = {"RMSE": None, "MAPE_pct": None, "note": "Prophet not installed"}

    # choose "best" by RMSE (lower is better)
    best_label = min([k for k in metrics if metrics[k]["RMSE"] is not None], key=lambda k: metrics[k]["RMSE"])
    best_preds = arima_preds if best_label=="ARIMA" else prophet_preds

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    # plots
    plot_actual_vs_pred(train, test, {"ARIMA": arima_preds, "Prophet": prophet_preds}, os.path.join(args.out_dir, "actual_vs_forecast.png"))
    plot_full_series_with_forecast(df, test, best_preds, best_label, os.path.join(args.out_dir, "series_with_best_forecast.png"))

    # save metrics JSON
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # save forecast CSV
    out_csv = pd.DataFrame({
        "date": test["date"],
        "actual": test["close"],
        "pred_ARIMA": arima_preds,
        "pred_Prophet": prophet_preds if prophet_preds is not None else [None]*len(test),
    })
    out_csv.to_csv(os.path.join(args.out_dir, "forecast_holdout.csv"), index=False)

    print("=== Forecast Complete ===")
    print(json.dumps(metrics, indent=2))
    print("Outputs saved in:", os.path.abspath(args.out_dir))

if __name__=="__main__":
    main()
