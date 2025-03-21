# Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Parameters and settings
ticker = "TSLA"
period = "25d"  # historical period to pull
interval = "1d"  # interval of data pulled
window = 20    # rolling window for moving average
threshold = 0.005  # threshold for decision making
capital = 10000  # starting capital in USD
transaction_fee_rate = 0.01  # 1% fee
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rolling_Mean', 'Spread']

# Load model
def get_model(path):
    # use libjar/pickle to load model
    print(f"loaded model from {path}")
    # return model

def get_data():
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise ValueError("No data retrived. Check network connection.")
    
    last_20_adj_close = data["Adj Close"].tail(20)
    rolling_mean = last_20_adj_close.rolling(window=window).mean().iloc[-1]

    yesterday = data.iloc[-1]
    yesterday["Rolling_Mean"] = rolling_mean
    yesterday["Spread"] = (yesterday["Adj Close"] - rolling_mean) / rolling_mean
    return yesterday[features]

# Run model evaluation
def evaluate(model, data):
    latest_features = data[features].values.reshape(1, -1)
    predicted_return = model.predict(latest_features)[0]
    spread = data['Spread']
    return predicted_return, spread

# Buy/Sell/Hold decision
def make_decision(predicted_return, spread):
    # Trading signal based on a mean reversion strategy:
    # - If price is below the moving average (spread is negative) and we predict an upward move,
    #   then it might be undervalued -> signal to Buy.
    # - If price is above the moving average (spread is positive) and we predict a downward move,
    #   then it might be overvalued -> signal to Sell.
    # - Otherwise, Hold.

    if predicted_return > threshold and spread < -threshold:
        decision = "Buy"
    elif predicted_return < -threshold and spread > threshold:
        decision = "Sell"
    else:
        decision = "Hold"

    print("\nTrading Decision:")
    print(f"Predicted Next Day Return: {predicted_return:.4f}")
    print(f"Spread (Deviation from mean): {spread:.4f}")
    print("Advice:", decision)

    return decision

# Simulate order details based on the decision.
def simulate_order(decision):
    # TODO: Implement a more robust way of tracking which orders have been made (i.e. csv or external file)
    # Number of shares held should be based on actual value

    if decision == "Buy":
        # Example: invest 10% of current capital (before transaction fees)
        buy_amount = capital * 0.1
        # Adjust for a 1% fee
        buy_amount_after_fee = buy_amount * (1 - transaction_fee_rate)
        print(f"Buy: ${buy_amount_after_fee:.2f} worth of shares.")
    elif decision == "Sell":
        # Example: assume you hold some shares; here, we'll simulate selling 10% of a placeholder holding.
        num_shares_held = 10  # This should come from your portfolio tracking system.
        shares_to_sell = max(int(num_shares_held * 0.1), 1)
        print(f"Sell: {shares_to_sell} shares (after applying transaction fees to proceeds).")
    else:
        print("Hold: No transaction will be made.")

    return 0

def main():
    model_path = ""

    # Use model with recent data to predict today's return and spread
    pred_return, spread = evaluate(get_model(model_path), get_data())
    # Make decision (Buy, Sell, Hold)
    decision = make_decision(pred_return, spread)
    simulate_order(decision)
    return 0
