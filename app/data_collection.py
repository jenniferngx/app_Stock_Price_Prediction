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
        return data  
    
    except Exception as e: 
        print(f"An error occurred: {e}")
        return pd.DataFrame()

if __name__ == "__main__": #Test function for data collection
    stock_data = fetch_stock_data("AAPL", "2021-01-01", "2021-06-31")
    stock_data.to_csv("data/apple_stock_data.csv", index=False)
    print("Data saved to apple_stock_data.csv")
