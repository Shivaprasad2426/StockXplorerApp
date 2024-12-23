import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import time



# Helper Functions
@st.cache_data
def load_data(file_path):
    """Load CSV data."""
    try:
        return pd.read_csv(file_path, parse_dates=['Date'])
    except Exception:
        st.error("Invalid file format or missing 'Date' column.")
        return None

@st.cache_data
def calculate_rmse(true_values, predictions):
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(true_values, predictions))

@st.cache_data
def fetch_alternate_live_data(ticker):
    """Fetch live stock data from an alternate API."""
    api_url = f"https://www.alphavantage.co/query"
    api_key = "ctkce79r01qntkqopbm0ctkce79r01qntkqopbmg"  # Replace with your actual API key
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": "1min",
        "apikey": api_key
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "Time Series (1min)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (1min)"], orient="index")
            df.reset_index(inplace=True)
            df.rename(columns={
                "index": "Datetime",
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            }, inplace=True)
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df["Close"] = df["Close"].astype(float)
            return df.sort_values("Datetime")
        else:
            st.error("API response did not include expected data.")
            return pd.DataFrame()
    else:
        st.error("Failed to fetch live data from the API.")
        return pd.DataFrame()

# App Setup
st.set_page_config(page_title="StockXplorer", layout="wide")
st.sidebar.title("StockXplorer")
st.sidebar.markdown("Navigate through different sections:")

tab = st.sidebar.radio("Go to:", ["Overview", "Time Series Models", "Live Data", "Advanced Features", "How to Use"])

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Stock CSV (with 'Date' and price columns)", type=["csv"])
data = load_data(uploaded_file) if uploaded_file else None

# Overview Section
if tab == "Overview":
    st.title("StockXplorer - Overview")
    if data is not None:
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)
        st.subheader("Data Preview")
        st.write(data.head())

        st.subheader("Line Chart")
        column_to_plot = st.selectbox("Select Column to Plot", data.columns)
        fig = px.line(data, x='Date', y=column_to_plot, title=f"{column_to_plot} Over Time")
        st.plotly_chart(fig)
    else:
        st.info("Upload a file to explore data.")

# Time Series Models Section
elif tab == "Time Series Models":
    st.title("Time Series Predictions")
    if data is not None and 'Close' in data.columns:
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]

        # ARIMA Model
        try:
            st.subheader("ARIMA Predictions")
            arima_model = ARIMA(train['Close'], order=(5, 1, 0))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=len(test))
            arima_rmse = calculate_rmse(test['Close'], arima_forecast)
            st.write(f"ARIMA RMSE: {arima_rmse:.2f}")
            st.line_chart(pd.DataFrame({'Actual': test['Close'], 'ARIMA': arima_forecast}).reset_index(drop=True))
        except Exception as e:
            st.error(f"ARIMA model error: {e}")

        # SARIMA Model
        try:
            st.subheader("SARIMA Predictions")
            sarima_model = SARIMAX(train['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit(disp=False)
            sarima_forecast = sarima_fit.forecast(steps=len(test))
            sarima_rmse = calculate_rmse(test['Close'], sarima_forecast)
            st.write(f"SARIMA RMSE: {sarima_rmse:.2f}")
            st.line_chart(pd.DataFrame({'Actual': test['Close'], 'SARIMA': sarima_forecast}).reset_index(drop=True))
        except Exception as e:
            st.error(f"SARIMA model error: {e}")

        # LSTM Model
        try:
            st.subheader("LSTM Predictions")
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(1, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')

            train_close = np.array(train['Close']).reshape(-1, 1)
            test_close = np.array(test['Close']).reshape(-1, 1)
            lstm_model.fit(train_close[:-1].reshape(-1, 1, 1), train_close[1:], epochs=10, batch_size=32)
            lstm_forecast = lstm_model.predict(test_close[:-1].reshape(-1, 1, 1)).flatten()
            lstm_rmse = calculate_rmse(test['Close'][1:], lstm_forecast)
            st.write(f"LSTM RMSE: {lstm_rmse:.2f}")
            st.line_chart(pd.DataFrame({'Actual': test['Close'][1:], 'LSTM': lstm_forecast}).reset_index(drop=True))
        except Exception as e:
            st.error(f"LSTM model error: {e}")
    else:
        st.warning("Please upload a dataset with a 'Close' column.")

# Live Data Section
elif tab == "Live Data":
    st.title("Live Stock Data and Predictions")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    if ticker:
        live_data = fetch_alternate_live_data(ticker)
        if not live_data.empty:
            if 'Close' in live_data.columns and 'Datetime' in live_data.columns:
                st.subheader("Live Data")
                st.write(live_data.tail())

                fig = px.line(live_data, x="Datetime", y="Close", title=f"Live Prices for {ticker}", labels={"Datetime": "Time", "Close": "Price"})
                st.plotly_chart(fig)

                # Simple Predictions
                st.subheader("Prediction")
                last_close = live_data['Close'].iloc[-1]
                max_days = len(live_data) if len(live_data) < 30 else 30
                future_days = st.slider("Days to Predict", min_value=1, max_value=max_days, value=7)
                prediction = [last_close * (1 + np.random.normal(0, 0.01)) for _ in range(future_days)]
                future_dates = pd.date_range(live_data['Datetime'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
                pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": prediction})
                st.write(pred_df)

                fig_pred = px.line(pred_df, x="Date", y="Predicted Close", title="Predicted Stock Prices")
                st.plotly_chart(fig_pred)
            else:
                st.error("Required columns 'Datetime' and 'Close' are missing from live data.")
        else:
            st.error("Failed to fetch live data. Try a different ticker.")

# Advanced Features
elif tab == "Advanced Features":
    st.title("Advanced Features")
    if data is not None:
        st.subheader("Correlation Matrix")
        uploaded_files = st.file_uploader("Upload Multiple CSV Files", accept_multiple_files=True)
        if uploaded_files:
            combined_data = {}
            for file in uploaded_files:
                stock_data = load_data(file)
                if stock_data is not None and 'Close' in stock_data.columns:
                    stock_name = file.name.split(".")[0]
                    combined_data[stock_name] = stock_data['Close']
            combined_df = pd.DataFrame(combined_data)
            fig, ax = plt.subplots()
            sns.heatmap(combined_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Upload data to access advanced features.")

# How to Use Section
elif tab == "How to Use":
    st.title("How to Use StockXplorer")
    st.markdown("""
    - **Upload CSV**: Provide historical stock data with 'Date' and price columns.
    - **Explore Features**: Navigate through different tabs to explore visualizations and models.
    - **Live Data**: Enter stock ticker to see real-time data and simple predictions.
    - **Advanced Features**: Analyze correlations across multiple stocks.
    """)
st.write("Refreshing data every 5 minutes...")
time.sleep(300)  # Pause for 5 minutes before fetching data again
