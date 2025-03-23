import argparse
import yfinance as yf
import os
import json
import pandas as pd
from datetime import datetime

ticker = "TSLA"
portfolio_filename = "portfolio.json"
transaction_fee_rate = 0.01  # 1% fee
capital = 10000  # starting capital in USD
current_price = 999999999999999.00

def get_current_price():
    tckr = yf.Ticker(ticker)
    price = tckr.info.get("regularMarketPrice")
    if price:
        return price
    else:
        return current_price

def get_10am_price():
    today = datetime.today().strftime('%Y-%m-%d')
    start_dt = f"{today} 09:30:00" # market open
    end_dt = f"{today} 16:00:00" # market close

    data = yf.download(ticker, start=start_dt, end=end_dt, interval="1m")
    data.index = pd.to_datetime(data.index)
    target_time = pd.to_datetime(f"{today} 10:00:00")

    try:
        price_at_10am = data.loc[target_time]
        print(f"Market price at 10am: {price_at_10am}")
        return price_at_10am
    except KeyError:
        print(f"No data found for {target_time}.")
        return current_price

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

# ====================== MAIN ======================

parser = argparse.ArgumentParser(description='execute buy order')

parser.add_argument('--capital', type=int, required=True, help='The amount of capital to be spent, including fee')
args = parser.parse_args()

buy_amount_after_fee = args.capital

current_price = get_10am_price()

buy_amount = buy_amount_after_fee / (1 - transaction_fee_rate)
shares_to_buy = int(buy_amount / current_price)

buy_amount_after_fee = shares_to_buy * current_price * (1 - transaction_fee_rate)

portfolio = get_portfolio_data()
portfolio["capital"] -= buy_amount_after_fee
portfolio["shares"] += shares_to_buy
update_portfolio(portfolio)
print(f"Buy: ${buy_amount_after_fee:.2f} worth of shares. ({shares_to_buy}) shares purchased.")