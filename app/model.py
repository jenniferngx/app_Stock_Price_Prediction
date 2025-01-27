from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import pandas as pd

def fit_arima(data, future_days, order = (5,1,0)):
    y = data["Close"].values
    model = ARIMA(y, order = order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps = future_days). tolist()
    return predictions

def cross_validate_arima(data, n_split = 5, order=(5,1,0)):
    fold_size = len(y) // n_split
    MSEs = []
    for i in range(1, n_split+1):
        train = y[:i*fold_size]
        test = y[i*fold_size : (i+1)*fold_size]
        predictions = fit_arima(data = train, future_days = len(test))
        MSE = mean_squared_error(test, predictions)
        MSEs.append(MSE)
    meanMSE = np.mean(MSEs)
    meanRMSE = np.sqrt(meanMSE)
    return meanRMSE

def cross_validate_lstm(data, n_split=5, lookback=5):
    """
        Params: 
            data (pd.DataFrame): stock historical prices
            n_split (int): number of splits for cross-validation
            lookback (int): number of past days to use for prediction
        Returns:
            metrics (dict): Average evaluation metrics across folds
    """
    y = data["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    y_scaled = scaler.fit_transform(y)
    fold_size = len(y_scaled) //n_split
    MSEs = []

    for i in range(1, n_split+1):                                                                                                                                                                                             
        train, test = y_scaled[:i*fold_size], y_scaled[i*fold_size: (i+1)*fold_size]
        if len(test) <= lookback:
            print(f"Skipping fold {i} due to insufficient test data.")
            continue

        x_train, y_train = [],[]
        for j in range(lookback, len(train)):
            x_train.append(train[j - lookback:j, 0])
            y_train.append(train[j, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        x_test, y_test = [], []
        for j in range(lookback, len(test)):
            x_test.append(test[j - lookback:j, 0])
            y_test.append(test[j, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
        model = Sequential() # Build and train LSTM model
        model.add(LSTM(50, return_sequences = True, input_shape = (lookback, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss = "mean_squared_error")
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        predictions = model.predict(x_test) # Predict and evaluate
        MSE = mean_squared_error(y_test, predictions)
        MSEs.append(MSE)
    
    meanMSE = np.mean(MSEs)
    meanRMSE = np.sqrt(meanMSE)
    return meanRMSE

def cross_validate_gb(data, n_split=5):
    """
    Params:
        data (pd.DataFrame): stock historical prices
        n_splits (int): number of splits for cross-validation
    Returns:
        metrics (dict): Average evaluation metrics across folds
    """
    data["Previous_Close"] = data["Close"].shift(1)
    x = data[["Previous_Close"]]
    y = data["Close"]
    tscv = TimeSeriesSplit()
    MSEs = []
    for train_idx, test_idx in tscv.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = xgb.XGBRegressor(objective = "reg:squarederror", n_estimators=100)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        MSE = mean_squared_error(y_test, predictions)
        MSEs.append(MSE)

    meanMSE = np.mean(MSEs)
    meanRMSE = np.sqrt(meanMSE)
    metrics = {
        "Gradient Boosting - Mean Squared Error": meanMSE,
        "Gradient Boosting - Root Mean Squared Error": meanRMSE,
    }
    return metrics



def fit_lstm(data, future_days, lookback=5):
    y = data["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range =(0,1))
    y_scaled = scaler.fit_transform(y)

    x = []
    y_labels = []
    for  i in range (lookback, len(y_scaled)):
        x.append(y_scaled[i-lookback: i,0])
        y_labels.append(y_scaled[i,0])
    x = np.array(x).reshape((len(x), lookback, 1))
    y_labels = np.array(y_labels)

    model = Sequential() # Fit LSTM model
    model.add(LSTM(50, return_sequences = True, input_shape = (lookback,1)))
    model.agoogdd(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer = "adam", loss = "mean_squared_error")
    model.fit(x, y_labels, epochs=10, batch_size=32, verbose=0)

    last_sequence = y_scaled[-lookback:]
    predictions = []
    for _ in range(future_days):
        prediction = model.predict(last_sequence.reshape(1, lookback, 1))[0,0]
        predictions.append(prediction)
        last_sequence = np.append(last_sequence[1:], prediction)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten().tolist()
    return predictions

def fit_gb(data, future_days):
    data["Previous_close"] = data["Close"].shift(1)
    data = data.dropna()
    x = data[["Previous_close"]].values
    y = data["Close"].values
    model = xgb.XGBRegressor(objective = "reg:squarederror", n_estimators=100)
    model.fit(x,y)
    predictions = []
    last_close = y[-1]
    for _ in range(future_days):
        prediction = model.predict([[last_close]])[0]
        predictions.append(prediction)
        last_close = prediction
    return predictions                                                                                                                                                                                                                                                                                                                                                                       

def predict_stock_price(data: pd.DataFrame, future_days: int):
    """
    Params:
        data (pd.DataFrame): stock historical prices
        future_days (int): number of future days to predict
    Returns:
        predictions of best model (ARIMA, LSTM, or Gradient Boosting)
    """
    arima_metrics = cross_validate_arima(data)
    lstm_metrics = cross_validate_lstm(data)
    gb_metrics = cross_validate_gb(data)

    RMSEs = {
        "ARIMA": arima_metrics.get("ARIMA - Root Mean Squared Error"),
        "LSTM": lstm_metrics.get("LSTM - Root Mean Squared Error"),
        "Gradient Boosting": gb_metrics.get("Gradient Boosting - Root Mean Squared Error")
    }
    best_model = min (RMSEs, key=RMSEs.get)
    print(f"Best Model is {best_model} with RMSE: {RMSEs[best_model]:.2f}")


def main():
    data = pd.read_csv("apple_stock_data.csv")
    best_model, future_predictions = predict_stock_price(data, future_days=5)
    print(f"Best model: {best_model}")
    print(f"Future predictions: {future_predictions}")

if __name__ == "__main__":
    main()