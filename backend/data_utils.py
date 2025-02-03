import yfinance as yf
import pandas as pd 

def fetch_stock_data(ticker:str, start_date:str, end_date:str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    data = stock.history(start = start_date, end=end_date)
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return pd.DataFrame() # return empty dataframe
        data.reset_index(inplace=True)

        print(f"Data for {ticker} from {start_date} to {end_date}")
        print(data.head()) 
        filepath =  f"data/{ticker}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        data.to_csv(filepath, index=False)
        print(f"Data for {ticker} from {start_date} to {end_date} saved to {filepath}")
        return data, filepath  
    
    except Exception as e: 
        print(f"An error occurred: {e}")
        return pd.DataFrame(), ""


if __name__ == "__main__": # Test function for data collection
    data, filepath = fetch_stock_data("AAPL", "2021-01-01", "2021-12-31")

