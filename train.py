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

#STEP 1
# TODO: This is currently importing data from the last year
# We could import this for training, but only the last X days are required for present day evaluation

print("Downloading historical data...")
data = yf.download(ticker, period=period, interval="1d")

# Ensure data is available
if data.empty:
    raise ValueError("No data was fetched. Check your ticker or network connection.")

# Import Training Data from CSV
# Load data from CSV
file_path = "TSLA.csv" 
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])

# STEP 2: Feature Engineering
# Calculate rolling (moving) mean of teh Adjusted Close price
data['Rolling_Mean'] = data['Adj Close'].rolling(window=window).mean()

# Calculate the "spread" - how far the current price is from the rolling mean (relative difference)
data['Spread'] = (data['Adj Close'] - data['Rolling_Mean']) / data['Rolling_Mean']

# Create a target price change: next day's return (percentage change)
data['Next_Day_Return'] = data['Adj Close'].shift(-1) / data['Adj Close'] - 1

# Remove rows with missing values (NaN) which result from rolling calculations and shifting
data.dropna(inplace=True) #TODO CHECK IF THIS RUINS THE SYSTEM
# I think this is used to remove the first/last row? Which wouldn't have data for the next-day-return
# We should test this to determine what this is doing exactly

# STEP 3: Prepare features and target Label for the ML Model
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rolling_Mean', 'Spread']
X = data[features]
y = data['Next_Day_Return']

# For time-series data, ensure we do not shuffle the data.
# Here we split based on an 80/20 ratio as an example.
# TODO: Check if we need to 80/20 split here. I'm not sure it's required since our test data will be real-time data
# Split data (keeping time-series continuity)
split_index = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# STEP 4: Train Random Forest Model
# Train model
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model to a file
model_filename = "tsla_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved to {model_filename}")

# Evaluate model
# TODO: This will not be required for the final version. Instead we will only be training the model.
# It may be worth testing so we have so evaluation metrics.
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Test Mean Squared Error: {mse: .6f}")

# STEP 5: Trading Decision and Simulation
# Run model evaluation

# We'll assume the "Latest" row represents today's data (to be used before 9:00 AM for submission).
# TODO: Write code to fetch only the data needed for today's estimation.
# TODO: Where's rolling mean here??? The model may not run if it's not included (calculate this)
# Make prediction for today's trading decision
latest = data.iloc[-1]
latest_features = latest[features].values.reshape(1, -1)
predicted_return = model.predict(latest_features)[0]
spread = latest['Spread']

# Buys/Sell/Hold decision

# Trading signal based on a mean reversion strategy:
# - If price is below moving average (spread is negative) and we predict an upward move,
# then it might be undervalued -> signal to Buy.
# - If price is above the moving average (spread is positive) and we predict a downward move,
# then it might be overvalued -> signal to Sell.
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

# Simulate order details based on the decision.

# TODO: Implement a more robust way of tracking which orders have been made (i.e. csv or external file)
# Number of shares held should be based on actual value

# Here we set simple rules for ordering sizing. In a full implementation, you would incorporate your current portfolio.
if decision == "Buy":
    # Example: invest 10% of current capital (before transaction fees)
    buy_amount = capital * 0.1
    # Adjust for a 1%fee
    buy_amount_after_fee = buy_amount * (1 - transaction_fee_rate)
    print(f"Buy: ${buy_amount_after_fee:.2f} worth of shares.")
elif decision == "Sell":
    # Example: assume you hold some shares; here, we'll simulate selling 10% of a placeholder holding.
    num_shares_held = 10  # This should come from your portfilio tracking system.
    shares_to_sell = max(int(num_shares_held * 0.1), 1)
    print(f"Sell: {shares_to_sell} shares (after applying transaction fees).")
else:
    print("Hold: No transaction will be made.")
