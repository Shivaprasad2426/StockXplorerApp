import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Helper Functions
@st.cache
def load_data(file_path):
    try:
        return pd.read_csv(file_path, parse_dates=['Date'])
    except ValueError:
        st.error("The uploaded file does not contain a 'Date' column.")
        return None

@st.cache
def calculate_rmse(true_values, predictions):
    return np.sqrt(mean_squared_error(true_values, predictions))

# App Title
st.title("StockXplorer")
st.sidebar.header("Navigation")
tab = st.sidebar.radio("Go to", ["Overview", "Time Series Models", "Live Data", "Advanced Features"])

# Load Data
uploaded_file = st.file_uploader("Upload a Stock CSV File (with 'Date' and price columns)", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None and 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)

        if tab == "Overview":
            st.header("Data Overview")
            if st.checkbox("Show Raw Data"):
                st.write(data.head())
            st.subheader("Line Chart")
            column_to_plot = st.selectbox("Select Column to Plot", data.columns)
            fig = px.line(data, x='Date', y=column_to_plot, title=f"{column_to_plot} Over Time")
            st.plotly_chart(fig)

        elif tab == "Time Series Models":
            st.header("Time Series Predictions")
            st.markdown("""
            - **LSTM**: Predicts based on sequence patterns using a neural network.
            - **ARIMA**: Best for non-seasonal data.
            - **SARIMA**: Adds seasonality to ARIMA.
            """)

            if 'Close' in data.columns:
                train_size = int(len(data) * 0.8)
                train, test = data[:train_size], data[train_size:]

                # ARIMA
                try:
                    st.subheader("ARIMA Predictions")
                    arima_model = ARIMA(train['Close'], order=(5, 1, 0))
                    arima_fit = arima_model.fit()
                    arima_forecast = arima_fit.forecast(steps=len(test))
                    arima_rmse = calculate_rmse(test['Close'], arima_forecast)
                    st.write(f"ARIMA RMSE: {arima_rmse:.2f}")
                    st.line_chart(pd.DataFrame({'Actual': test['Close'], 'ARIMA': arima_forecast}).reset_index(drop=True))
                except Exception as e:
                    st.error(f"ARIMA model failed: {e}")

                # SARIMA
                try:
                    st.subheader("SARIMA Predictions")
                    sarima_model = SARIMAX(train['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    sarima_fit = sarima_model.fit(disp=False)
                    sarima_forecast = sarima_fit.forecast(steps=len(test))
                    sarima_rmse = calculate_rmse(test['Close'], sarima_forecast)
                    st.write(f"SARIMA RMSE: {sarima_rmse:.2f}")
                    st.line_chart(pd.DataFrame({'Actual': test['Close'], 'SARIMA': sarima_forecast}).reset_index(drop=True))
                except Exception as e:
                    st.error(f"SARIMA model failed: {e}")

                # LSTM
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
                    st.error(f"LSTM model failed: {e}")


      

        

        elif tab == "Advanced Features":
            st.header("Advanced Features")
            st.subheader("Correlation Matrix")
            uploaded_files = st.file_uploader("Upload Multiple Stock CSV Files", accept_multiple_files=True)
            if uploaded_files:
                combined_data = {}
                for file in uploaded_files:
                    stock_data = load_data(file)
                    if stock_data is not None and 'Close' in stock_data.columns:
                        stock_name = file.name.split(".")[0]
                        combined_data[stock_name] = stock_data['Close']
                combined_df = pd.DataFrame(combined_data)
                sns.heatmap(combined_df.corr(), annot=True, cmap="coolwarm")
                st.pyplot()

# Time Series Predictions Section
st.header("Future Predictions")

if 'Close' in data.columns:
    # Train-test split
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    # User input for prediction days
    future_days = st.slider("Select Number of Days to Predict", min_value=1, max_value=365, value=30)

    # ARIMA Predictions
    try:
        st.subheader("ARIMA Future Predictions")
        arima_model = ARIMA(train['Close'], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=future_days)

        # Create future dates
        last_date = data['Date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

        # Combine dates and predictions
        arima_df = pd.DataFrame({'Date': future_dates, 'ARIMA Prediction': arima_forecast})
        st.write("ARIMA Future Predictions:")
        st.write(arima_df)

        # Plot ARIMA predictions
        fig = px.line(arima_df, x='Date', y='ARIMA Prediction', title="ARIMA Future Predictions")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"ARIMA prediction failed: {e}")

    # SARIMA Predictions
    try:
        st.subheader("SARIMA Future Predictions")
        sarima_model = SARIMAX(train['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_fit = sarima_model.fit(disp=False)
        sarima_forecast = sarima_fit.forecast(steps=future_days)

        # Combine dates and predictions
        sarima_df = pd.DataFrame({'Date': future_dates, 'SARIMA Prediction': sarima_forecast})
        st.write("SARIMA Future Predictions:")
        st.write(sarima_df)

        # Plot SARIMA predictions
        fig = px.line(sarima_df, x='Date', y='SARIMA Prediction', title="SARIMA Future Predictions")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"SARIMA prediction failed: {e}")

    # LSTM Predictions
    try:
        st.subheader("LSTM Future Predictions")

        # Prepare data for LSTM
        train_close = np.array(train['Close']).reshape(-1, 1)
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(train_close[:-1].reshape(-1, 1, 1), train_close[1:], epochs=10, batch_size=32)

        # Generate future predictions
        last_close = train_close[-1]
        lstm_predictions = []
        for _ in range(future_days):
            next_close = lstm_model.predict(last_close.reshape(1, 1, 1)).flatten()[0]
            lstm_predictions.append(next_close)
            last_close = np.array([next_close])

        # Combine dates and predictions
        lstm_df = pd.DataFrame({'Date': future_dates, 'LSTM Prediction': lstm_predictions})
        st.write("LSTM Future Predictions:")
        st.write(lstm_df)

        # Plot LSTM predictions
        fig = px.line(lstm_df, x='Date', y='LSTM Prediction', title="LSTM Future Predictions")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"LSTM prediction failed: {e}")
else:
    st.error("The dataset must contain a 'Close' column for predictions.")






st.header("Live Stock Data")
ticker = st.text_input("Enter Stock Ticker", "AAPL")

if ticker:
    # Download live data
    live_data = yf.download(ticker, period="1d", interval="1m")

    # Check if the data is valid
    if live_data.empty:
        st.error("No data available for the given ticker and time period. Please try a different ticker or adjust the time period.")
    else:
        # Reset index to ensure 'Datetime' is a column
        live_data.reset_index(inplace=True)

        # Validate required columns
        if 'Close' not in live_data.columns:
            st.error("'Close' column is missing from the data.")
        elif live_data['Close'].isnull().all():  # Corrected this condition
            st.error("'Close' column contains only NaN values. Cannot process the data.")
        else:
            # Ensure columns are properly formatted
            live_data['Close'] = pd.to_numeric(live_data['Close'], errors='coerce')
            live_data['Datetime'] = pd.to_datetime(live_data['Datetime'], errors='coerce')

            # Remove rows with invalid data
            live_data.dropna(subset=['Datetime', 'Close'], inplace=True)

            # Plot the data
            fig = px.line(
                live_data,
                x="Datetime",
                y="Close",
                title=f"Live Prices for {ticker}",
                labels={"Datetime": "Time", "Close": "Price"}
            )
            st.plotly_chart(fig)
