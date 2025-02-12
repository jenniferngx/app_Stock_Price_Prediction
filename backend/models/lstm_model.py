from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from itertools import product
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def forecast_LSTM(data, col, daysAhead):
    # Ensure Date column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])  
    data.set_index('Date', inplace=True) 

    # Find best parameters for LSTM model
    window = find_window(data=data, col=col, daysAhead=daysAhead)

    # Extract target column and scale data
    y = data[col].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    y_scaled = scaler.fit_transform(y)

    train = y_scaled[:-window]

    # Create training data
    x_train, y_train = [], []
    for i in range(window, len(train)):
        x_train.append(train[i - window:i, 0])
        y_train.append(train[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    print(f"✅ Training data prepared: {x_train.shape} => Train size: {len(train)}")

    # Define LSTM model
    print("🚀 Compiling and training LSTM model...")
    model = Sequential()
    model.add(LSTM(120, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss=Huber(delta=1.0))

    # Use EarlyStopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, callbacks=[early_stop])

    predictions = []
    last_window = y_scaled[-window: ].reshape(1, window, 1)
    for _ in range(daysAhead):
        next_pred  = model.predict(last_window)[0,0]
        predictions.append(next_pred)
        last_window = np.append(last_window[:, 1:, :], [[[next_pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=data.index[-1], periods=daysAhead+1, freq='B')[1:]
    print(f"✅ Predictions: {predictions.flatten()}")

    plt.figure(figsize = (16,8))
    plt.plot(data.index, scaler.inverse_transform(y_scaled), label = 'train', color = 'b')
    plt.plot(future_dates, predictions.flatten(), label = 'predict', color = 'r')
    plt.title(f"LSTM Model - {daysAhead}-Day Rolling Predictions")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("lstm_forecast.png")
    plt.close()

    return predictions.flatten()


def fit_LSTM(data, col, daysAhead):
    # Find best parameters for LSTM model
    window = find_window(data=data, col=col, daysAhead=daysAhead)

    # Extract target column and scale data
    y = data[col].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    y_scaled = scaler.fit_transform(y)

    train_size = int(len(data) *0.8)
    train, test = y_scaled[:train_size], y_scaled[train_size - window:]

    # Create training data
    x_train, y_train = [], []
    for i in range(window, len(train)):
        x_train.append(train[i-window:i, 0])
        y_train.append(train[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    print(f"✅ Training data prepared: {x_train.shape} => Train size: {len(train)}")

    # Create testing data
    x_test, y_test = [], []
    for j in range(window, len(test)):
        x_test.append(test[j -window:j, 0])
        y_test.append(test[j, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(f"✅ Testing data prepared: {x_test.shape} => Test size: {len(test)}")

    # Define LSTM model
    print("🚀 Compiling and training LSTM model...")
    model = Sequential()
    model.add(LSTM(120, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss=Huber(delta=1.0))

    # Use EarlyStopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, callbacks=[early_stop])



    # Make predictions & Calculate MSE
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    MSE = mean_squared_error(y_test, predictions)
    print("The rmse value is:", {float(np.sqrt(MSE))})

    # Plotting
    train_plot = scaler.inverse_transform(train)
    plt.figure(figsize = (16,8))
    plt.plot(data.index[:train_size], train_plot, label = 'train', color = 'b')
    plt.plot(data.index[train_size:], y_test.flatten(), label = 'true', color = 'r')
    plt.plot(data.index[train_size:], predictions.flatten(), label ='predict', color = 'k')
    plt.title('LSTM Model - Predictions vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("lstm_model.png")
    plt.close()
    
    return predictions.flatten()


def evaluate_window(data, col, window):
    # Extract, scale data & split into train and test sets
    y = data[col].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    y_scaled = scaler.fit_transform(y)
    train_size = int(len(data) * 0.8)
    train = y_scaled[:train_size]
    test = y_scaled[train_size:]   

    if len(test) <= window:
        raise ValueError(f"Test set size ({len(test)}) is smaller than the window size ({window}).")

    # Create training sequences
    x_train, y_train = [], []
    for i in range(window, len(train)):
        x_train.append(train[i - window:i, 0])
        y_train.append(train[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # Create validation sequences
    x_test, y_test = [], []
    for j in range(window, len(test)):
        x_test.append(test[j - window:j, 0])
        y_test.append(test[j, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Build model
    model = Sequential()
    model.add(LSTM(120, input_shape=(x_train.shape[1], 1), return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        x_train, y_train, 
        validation_data = (x_test, y_test),
        batch_size=64, 
        epochs= 50, 
        verbose=0,
        callbacks=[early_stop])
    validation_loss = history.history['val_loss'][-1]
    return validation_loss

def find_window(data, col, daysAhead):
    if daysAhead <7:
        windows = [7,14]
    elif daysAhead >=7 & daysAhead <30:
        windows = [21,30]
    else: windows = [60,90]

    best_validation_loss = float('inf')
    best_window = None
    
    for window in windows:
        print(f"\nTesting: Window={window}")
        try:
            # Evaluate LSTM with current parameter set
            validation_loss = evaluate_window(data, col, window)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_window = window
                print(f"\nTesting: Window={window} - Current best with validation loss: {best_validation_loss:.4f}")
            else: 
                print(f"\nTesting: Window={window}")
        except ValueError as e:
            # Handle test set size issues or other errors
            print(f"Skipping combination Window={window} due to error: {e}")
            continue
    print(f"✅ Best LSTM parameters: Windows={best_window} with validation loss: {best_validation_loss:.4f}")   
    return best_window

# Testing functionality

"""
data = pd.read_csv("data/BB_20220211_20250210.csv")
fit_LSTM(data, col="Close", daysAhead=20)
forecast_LSTM(data, col="Close", daysAhead=20)
"""
"""
data = pd.read_csv("data/NVDA_20240211_20250211.csv")
forecast_LSTM(data, col="Close", daysAhead=20)

"""