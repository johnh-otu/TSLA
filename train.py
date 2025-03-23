#Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Parameters and settings
ticker = "TSLA"
period = "1y" # historical period to pull
window = 20    # rolling window for moving average
threshold = 0.005  # threshold for decision making
capital = 10000  # starting capital in USD
transaction_fee_rate = 0.01  # 1% fee

# Get Data
def download_data():
    print("Downloading Historical Data...")
    data = yf.download(ticker, period=period, interval="1d")
    # Ensure data is available
    if data.empty:
        raise ValueError("No data was fetched. Check your ticker or network connection.")
    return data
def import_data():
    print("Importing Training Data...")
    # Load data from CSV
    file_path = "TSLA.csv" 
    data = pd.read_csv(file_path)
    return data

# Train model
def train_model(X_train, y_train):
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Save trained model to a file
def save_model(model):
    model_filename = "tsla_model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filename}")
    return 0

# Feature Engineering
def feature_eng(data):
    print("Preparing Data...")
    
    # Convert 'Date' column to datetime format
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

    # Calculate rolling (moving) mean of teh Adjusted Close price
    data['Rolling_Mean'] = data['Adj Close'].rolling(window=window).mean()

    # Calculate the "spread" - how far the current price is from the rolling mean (relative difference)
    data['Spread'] = (data['Adj Close'] - data['Rolling_Mean']) / data['Rolling_Mean']

    # Create a target price change: next day's return (percentage change)
    data['Next_Day_Return'] = data['Adj Close'].shift(-1) / data['Adj Close'] - 1

    # Remove rows with missing values (NaN) which result from rolling calculations and shifting
    data.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rolling_Mean', 'Spread']
    X = data[features]
    y = data['Next_Day_Return']
    return X, y

# ====================== MAIN ======================
data = import_data()
X, y = feature_eng(data)
model = train_model(X, y)
save_model(model)