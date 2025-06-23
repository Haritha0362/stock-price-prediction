import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

st.title("📈 Stock Price Prediction and Recommendation System")

# Sidebar Inputs
st.sidebar.header("User Input")
company = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, MSFT):", "AAPL").upper()
start_date = st.sidebar.date_input("Start date", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.today())
run_lstm = st.sidebar.checkbox("Include LSTM (slower)", value=False)

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    data.reset_index(inplace=True)
    return data

if company:
    st.subheader(f"Fetching data for {company}...")
    data = load_data(company, start_date, end_date)

    if data.empty:
        st.error("❌ No data found. Please check the stock ticker or date range.")
        st.stop()

    # Preprocessing
    df = data[['Date', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    features = ['Close', 'MA7', 'MA20', 'MA50']
    X = df[features]
    y = df['Target']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Training models..."):
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        lr_r2 = r2_score(y_test, lr_pred)

        # Optional: LSTM
        if run_lstm:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[features + ['Target']])
            seq_length = 10
            X_lstm, y_lstm = [], []
            for i in range(seq_length, len(scaled_data)):
                X_lstm.append(scaled_data[i-seq_length:i, :-1])
                y_lstm.append(scaled_data[i, -1])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            split = int(0.8 * len(X_lstm))
            X_lstm_train, y_lstm_train = X_lstm[:split], y_lstm[:split]
            X_lstm_test, y_lstm_test = X_lstm[split:], y_lstm[split:]

            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])),
                LSTM(50),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=32, verbose=0)

            lstm_pred_scaled = lstm_model.predict(X_lstm_test, verbose=0)
            # Proper inverse transform using padding for unscaled features
            dummy = np.zeros((lstm_pred_scaled.shape[0], scaled_data.shape[1]-1))
            lstm_pred = scaler.inverse_transform(np.hstack((dummy, lstm_pred_scaled)))[:, -1]
            y_lstm_true = scaler.inverse_transform(np.hstack((dummy, y_lstm_test.reshape(-1, 1))))[:, -1]
            lstm_rmse = np.sqrt(mean_squared_error(y_lstm_true, lstm_pred))
            lstm_r2 = r2_score(y_lstm_true, lstm_pred)

    # Display Metrics
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("RF RMSE", f"{rf_rmse:.2f}", f"R²: {rf_r2:.3f}")
    col2.metric("LR RMSE", f"{lr_rmse:.2f}", f"R²: {lr_r2:.3f}")
    if run_lstm:
        col3.metric("LSTM RMSE", f"{lstm_rmse:.2f}", f"R²: {lstm_r2:.3f}")

    # Comparison Bar Chart
    st.subheader("📉 RMSE and R² Comparison")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    models = ['Random Forest', 'Linear Regression']
    rmse_values = [rf_rmse, lr_rmse]
    r2_values = [rf_r2, lr_r2]
    if run_lstm:
        models.append('LSTM')
        rmse_values.append(lstm_rmse)
        r2_values.append(lstm_r2)

    ax[0].bar(models, rmse_values, color=['blue', 'green', 'orange'][:len(models)])
    ax[0].set_title("RMSE")
    ax[1].bar(models, r2_values, color=['blue', 'green', 'orange'][:len(models)])
    ax[1].set_title("R² Score")
    st.pyplot(fig)

    # Plot RF Prediction
    st.subheader(f"{company} - Last 60 Days Prediction (Random Forest)")
    last_n = 60
    plot_dates = df['Date'].iloc[-len(y_test):].values[-last_n:]
    actual_prices = y_test.values[-last_n:]
    predicted_prices = rf_pred[-last_n:]

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(plot_dates, actual_prices, label='Actual', color='blue', marker='o')
    ax2.plot(plot_dates, predicted_prices, label='RF Predicted', color='orange', marker='o')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# --------------------------
# Recommendation
# --------------------------
st.subheader("💡 Recommendation")

latest_features = df[features].iloc[-1].values.reshape(1, -1)
next_day_price = rf_model.predict(latest_features)[0]  # Ensure scalar
current_price = df['Close'].iloc[-1]  # Scalar

# Ensure both are floats
change = float((next_day_price - current_price) / current_price * 100)

st.write(f"Next Day Predicted Price (Random Forest): **${next_day_price:.2f}**")

if change > 5:
    st.success("✅ Recommendation: **Buy** (Expected ↑ more than 5%)")
elif change < -5:
    st.error("❌ Recommendation: **Sell** (Expected ↓ more than 5%)")
else:
    st.info("⏳ Recommendation: **Hold** (No major change expected)")
