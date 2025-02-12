import yfinance as yf
import pandas as pd 
from datetime import datetime, timedelta

def fetch_stock_data(ticker:str, start_date:str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    today = datetime.now().strftime('%Y-%m-%d')

    #data = stock.history(start = start_date, end=today)
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=today)
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return pd.DataFrame() # return empty dataframe
        data.reset_index(inplace=True)


        print(f"Data for {ticker} from {start_date} to {today}")
        print(data.head()) 
        return data
    
    except Exception as e: 
        print(f"An error occurred: {e}")
        return pd.DataFrame(), ""

def store_stock_data(ticker, df, start_date):
    today = datetime.now().strftime('%Y%m%d')

    filepath =  f"data/{ticker}_{start_date.replace('-','')}_{today}.csv"
    df.to_csv(filepath, index=False)
    
    print(f"Data for {ticker} from {start_date} to {today} saved to {filepath}")
    return filepath


if __name__ == "__main__": # Test data collection
    data = fetch_stock_data("NVDA", "2024-02-11")
    filepath = store_stock_data("NVDA", data, "2024-02-11")

