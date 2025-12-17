import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
Sequential = tf.keras.models.Sequential
EarlyStopping = tf.keras.callbacks.EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout

def generate_time_series(n_points=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    start_date = datetime(2020,1,1)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    noise = np.random.normal(0,5,n_points)
    trend = np.linspace(50,120,n_points)
    season = 15*np.sin(np.arange(n_points)*2*np.pi/365)
    values = noise + trend + season
    return pd.DataFrame({"date":dates, "value":values})

def compute_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = sqrt(mean_squared_error(true, pred))
    mask = true != 0
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    return {"MAE":mae, "RMSE":rmse, "MAPE":mape}

def safe_sarimax_fit(y, order, seasonal_order, maxiter=20):
    try:
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(maxiter=maxiter, disp=False)
        return res
    except:
        return None

def create_sequences(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default=None)
    parser.add_argument('--n_points', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--forecast_horizon', type=int, default=90)
    parser.add_argument('--window', type=int, default=30)
    args = parser.parse_args()

    if args.input_csv and os.path.exists(args.input_csv):
        df = pd.read_csv(args.input_csv, parse_dates=['date'])
    else:
        df = generate_time_series(args.n_points, seed=args.seed)

    df = df.sort_values('date').set_index('date')
    try:
        df = df.asfreq('D')
    except:
        pass

    if args.plot:
        df.plot(figsize=(12,5))
        plt.title("Исходный временной ряд")
        plt.show()

    series = df['value']

    try:
        sd = seasonal_decompose(series, model='additive', period=365)
        if args.plot:
            sd.plot()
            plt.show()
    except:
        pass

    if args.plot:
        plot_acf(series, lags=40)
        plt.show()
        plot_pacf(series, lags=40)
        plt.show()

    y = df['value'].dropna()

    best_res = None
    best_aic = np.inf

    for p in range(2):
        for d in range(2):
            for q in range(2):
                order = (p,d,q)
                seasonal_order = (0,0,0,365)
                res = safe_sarimax_fit(y, order, seasonal_order)
                if res is None:
                    continue
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_res = res

    sar_res = best_res
    horizon = args.forecast_horizon
    sar_pred = sar_res.get_forecast(steps=horizon)
    sar_mean = sar_pred.predicted_mean
    sar_ci = sar_pred.conf_int()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df['value'].values.reshape(-1,1))

    window = args.window
    X, y_lstm = create_sequences(scaled.flatten(), window)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(0.8*len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_lstm[:split], y_lstm[split:]

    model = Sequential()
    model.add(LSTM(64, input_shape=(window,1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60,
              batch_size=32, callbacks=[es], verbose=0)

    last_window = scaled.flatten()[-window:]
    lstm_preds = []
    for _ in range(horizon):
        x_in = last_window.reshape(1,window,1)
        p = model.predict(x_in, verbose=0).flatten()[0]
        lstm_preds.append(p)
        last_window = np.roll(last_window, -1)
        last_window[-1] = p
    lstm_preds = scaler.inverse_transform(np.array(lstm_preds).reshape(-1,1)).flatten()

    if args.plot:
        future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon)
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['value'], label='observed')
        plt.plot(future_index, sar_mean, label='SARIMAX')
        plt.plot(future_index, lstm_preds, label='LSTM')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
