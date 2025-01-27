import pandas as pd

def add_features(data_path):
    data = pd.read_csv(data_path)
    data["Change"] = data["Close"].diff().fillna(0)
    data.to_csv(data_path, index=False)

add_features("data/apple_stock_data.csv")
