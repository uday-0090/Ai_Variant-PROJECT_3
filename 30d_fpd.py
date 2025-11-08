import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

#Mounting Google drive to load the dataset
from google.colab import drive

drive.mount('/content/drive')

file_path = '/content/drive/My Drive/Datasets/PJMW_hourly.csv'
df = pd.read_csv(file_path)

# --- Streamlit UI ---
st.title("🔌 30-Day Power Demand Forecast using LightGBM")
st.write("This app forecasts the hourly PJMW power demand for the next 30 days using a trained LightGBM model.")


# --- Load & Prepare Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly.csv", parse_dates=['Datetime'])
    df.rename(columns={'Datetime': 'date&time'}, inplace=True)
    df.set_index('date&time', inplace=True)

    # Use only the last 4 years
    df = df[df.index >= df.index.max() - pd.DateOffset(years=4)]

    # Fill missing values if any
    df['PJMW_MW'] = df['PJMW_MW'].fillna(method='ffill').astype(int)
    return df


df = load_data()


# --- LightGBM Forecast Function ---
def forecast_next_30_days(df, target_col='PJMW_MW', lag=168, future_hours=720):
    df_copy = df[[target_col]].copy()

    # Standardize the target
    scaler_y = StandardScaler()
    df_copy[target_col] = scaler_y.fit_transform(df_copy)

    # Create lag features
    for i in range(1, lag + 1):
        df_copy[f'lag_{i}'] = df_copy[target_col].shift(i)

    df_lagged = df_copy.dropna()
    X = df_lagged.drop(columns=[target_col])
    y = df_lagged[target_col]

    # Train model
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X, y)

    # Start forecasting
    history = list(scaler_y.transform(df[[target_col]]).flatten())
    future_scaled = []

    for _ in range(future_hours):
        input_lags = history[-lag:]
        input_df = pd.DataFrame([input_lags], columns=[f'lag_{i}' for i in range(1, lag + 1)])
        pred_scaled = model.predict(input_df)[0]
        history.append(pred_scaled)
        future_scaled.append(pred_scaled)

    # Inverse transform forecast
    future_forecast = scaler_y.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()

    # Create future datetime index
    last_timestamp = df.index[-1]
    future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=future_hours, freq='H')
    future_series = pd.Series(future_forecast, index=future_index, name='Forecast_PJMW_MW')

    return future_series


# --- Run Forecast ---
with st.spinner("Forecasting next 30 days..."):
    forecast_series = forecast_next_30_days(df)

# --- Show Result ---
st.success("Forecast complete!")
st.write(forecast_series)

# Optional download
csv = forecast_series.reset_index().rename(columns={"index": "DateTime"}).to_csv(index=False)
st.download_button("Download Forecast CSV", data=csv, file_name="30_day_forecast.csv", mime="text/csv")
