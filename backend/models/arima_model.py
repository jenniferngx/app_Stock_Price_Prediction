import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def forecast_ARIMA(data, col, steps):
    train = data[col].values

    (p,d,q) = find_params_ARIMA(data, col="Close", P=range(0,11), D=range(0,3), Q=(0,11))

    print("Fitting ARIMA model...")
    model = sm.tsa.arima.ARIMA(train, order=(p,d,q)).fit()

    print("Making predictions with fitted model...")
    predictions = model.predict(start = len(train), end = len(train)+steps-1, dynamic = False)
    print(predictions)

    dates = pd.to_datetime(data["Date"])
    train_dates = dates[:len(train)] 
    last_date = train_dates.iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
    
    # Plotting predictions
    plt.figure(figsize = (10,8))
    plt.plot(train_dates, train, color='k', label='train')  # Training data
    plt.plot(future_dates, predictions, color='b', label='predict')  # Predictions
    plt.title(f"ARIMA model prediction for next {steps} days")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("plots/arima_forecast.png")
    plt.close()


def fit_ARIMA(data, col):
    x = data[col].values
    (p,d,q) = find_params_ARIMA(data, col="Close", P=range(0,11), D=range(0,3), Q=(0,11))

    test_size = int(len(x) * 0.2)
    train = x[:-test_size]
    test = x[-test_size:]

    print("Fitting ARIMA model...")
    model = sm.tsa.arima.ARIMA(train, order=(p,d,q)).fit()

    print("Making predictions with fitted model...")
    pred = model.predict(start = len(train), end = len(x)-1, dynamic = False)

    # Plotting
    train_pred = train
    train_pred = np.concatenate((train_pred, pred), axis=None)
    plt.figure(figsize = (10,8))
    plt.plot(train_pred, color = 'b', label = 'predict')
    plt.plot(x, color = 'r', label = 'real')
    plt.plot(train, label = 'train', color = 'k')
    plt.title("ARIMA Model - Predictions vs Actual Prices")
    plt.legend()
    plt.savefig("plots/arima_model.png")
    plt.close()

    # rmse
    print("The rmse value is:", round(np.sqrt(mean_squared_error(pred,test)),0))
    return np.sqrt(mean_squared_error(pred,test))


def evaluate_arima(data, col, p, d, q):
    train_size = int(len(data) * 0.8)
    train,test = data[col].values[:train_size], data[col].values[train_size:]
    model = sm.tsa.arima.ARIMA(train, order=(p,d,q)).fit()
    pred = model.predict(start=len(train), end= len(train)+len(test) -1, dynamic=False)
    rmse = np.sqrt(mean_squared_error(test,pred))
    return rmse

def find_params_ARIMA(data, col, P, D, Q):
    best_rmse = np.inf
    best_params = None
    if is_stationary(data[col].values):
        D = [0]
    for p, d, q in itertools.product(P, D, Q):
        try: 
            rmse = evaluate_arima(data, col, p, d, q)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (p, d, q)
                print(f"New best RMSE: {best_rmse:.4f} with params (p,d,q): ({p},{d},{q})")
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Skipping (p,d,q): ({p},{d},{q}) due to error: {e}")
            continue
    print(f"Best ARIMA parameters: (p, d, q) = ({p}, {d}, {q}) with RMSE: {best_rmse}")
    return best_params


def is_stationary(series):
    result = adfuller(series)
    if result[1] <= .05:
        return True # Series is stationary 
    else: 
        return False # Series is non-stationary -> needs differencing


def main(): # For testing purpose
    data = pd.read_csv("backend/data/AAPL_20210101_20211231.csv")

    print("\nTesting fit_ARIMA function...") 
    fit_ARIMA(data, col="Close")

    #print("\nTesting forecast_ARIMA function...")
    #forecast_ARIMA(data, col="Close", steps = 7)



main()

"""
def cv_ARIMA(data, col, p, d, q, k=5):
    series = data[col].values
    tscv = TimeSeriesSplit(n_splits = k)
    RMSEs = []

    for split, (train_idx, test_idx) in enumerate(tscv.split(series), start=1):
        train, test = series[train_idx], series[test_idx]
        model = sm.tsa.arima.ARIMA(train, order = (p,d,q)).fit()
        preds = model.forecast(steps=len(test))

        if (len(preds) != len(test)):
            print(f"Split {split}: Prediction size and test size mismatch!")
            continue

        # Plotting for each split
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(train)), train, label="Train Data")
        plt.plot(range(len(train), len(train) + len(test)), test, label="Test Data", color="orange")
        plt.plot(range(len(train), len(train) + len(preds)), preds, label="Predictions", color="green")
        plt.title(f"ARIMA Predictions for Split {split}")
        plt.legend()
        plt.savefig(f"split_{split}_arima.png")
        plt.close()
        
        # Compute 
        RMSE = np.sqrt(mean_squared_error(test, preds))
        RMSEs.append(RMSE)
        print(f"RMSE for split {split}: {RMSE} - Mean prediction for current split: {np.mean(preds)}")
    return (np.mean(RMSEs))
"""