# Import Libraries
import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
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
portfolio_filename = "portfolio.json"
current_price = 999999999999999.99
model_path = "tsla_model.pkl" 
sell_ratio = 0.4 # sell 40% of shares (originally 10%)
buy_ratio = 0.1 # buy with 10% of capital

# Load model
def get_model(path):
    model = pickle.load(open(model_path, "rb"))
    print(f"loaded model from {path}")
    return model

def get_data():
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise ValueError("No data retrived. Check network connection.")
    
    last_20_adj_close = data["Adj Close"].tail(window)
    rolling_mean = last_20_adj_close.rolling(window=window).mean().iloc[-1]

    yesterday = data.iloc[-1]
    yesterday["Rolling_Mean"] = rolling_mean
    yesterday["Spread"] = (yesterday["Adj Close"] - rolling_mean) / rolling_mean
    return yesterday[features]

# Run model prediction
def predict(model, data):
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

def get_portfolio_data():
    
    default = {"shares": 0, "capital": capital}
    portfolio_data = {}

    # Check if portfolio file exists
    #   - If exists -> import data
    #   - If not exists -> generate new file and use default data
    if not os.path.exists(portfolio_filename):
        return update_portfolio(default)
    else:
        try:
            with open(portfolio_filename, "r") as fp:
                portfolio_data = json.load(fp)
        except json.JSONDecodeError:
            # if corrupt or empty, reset to default
            portfolio_data = default
            update_portfolio(portfolio_data)
        
        # check all expected keys exist
        for key, val in default.items():
            if key not in portfolio_data:
                portfolio_data[key] = val
    
    return portfolio_data

def update_portfolio(portfolio):
    with open(portfolio_filename, "w") as fp:
            json.dump(portfolio, fp, indent=4)
    return portfolio

# Simulate order details based on the decision.
def simulate_order(decision, current_price):

    portfolio = get_portfolio_data()

    if decision == "Buy":
        # Example: invest 10% of current capital (before transaction fees)
        buy_amount = capital * buy_ratio
        # Adjust for a 1% fee
        buy_amount_after_fee = buy_amount * (1 - transaction_fee_rate)
        shares_to_buy = int(buy_amount / current_price)

        if shares_to_buy == 0:
            return "Hold: No transaction will be made."
        else:
            buy_amount = shares_to_buy * current_price
            buy_amount_after_fee = buy_amount * (1 - transaction_fee_rate)
            portfolio["capital"] -= buy_amount_after_fee
            portfolio["shares"] += shares_to_buy
            update_portfolio(portfolio)
            return f"Buy: ${buy_amount_after_fee:.2f} worth of shares."
    elif decision == "Sell":
        # Simulate selling 10% of a placeholder holding.
        num_shares_held = portfolio["shares"]

        if (num_shares_held < 1):
            return "Hold: No transaction will be made."

        shares_to_sell = max(int(num_shares_held * sell_ratio), 1)
        return f"Sell: {shares_to_sell} shares (after applying transaction fees to proceeds)."
    else:
        return "Hold: No transaction will be made."

def get_current_price():
    tckr = yf.Ticker(ticker)
    price = tckr.info.get("regularMarketPrice")
    if price:
        return price
    else:
        return current_price

# ====================== MAIN ======================

# Use model with recent data to predict today's return and spread
pred_return, spread = predict(get_model(model_path), get_data())
# Make decision (Buy, Sell, Hold)
decision = make_decision(pred_return, spread)
# Get current price for buy/sell orders
if decision != "Hold":
    current_price = get_current_price()
print(simulate_order(decision, current_price))
