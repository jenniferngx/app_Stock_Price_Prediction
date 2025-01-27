from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def LSTM_model(data, col, window):
    # Extract target column and scale data
    y = data[col].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    y_scaled = scaler.fit_transform(y)

    # Split into training and testing sets
    train_size = int(len(data) *0.8)
    train = y_scaled[:train_size]
    
    # Create training sequences
    x_train, y_train = [], []
    for i in range(window, len(train)):
        x_train.append(train[i-window:i, 0])
        y_train.append(train[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(120, input_shape=(x_train.shape[1], 1), return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=128, epochs=50)

    # Prepare testing data
    test = y_scaled[train_size - window:]
    x_test, y_test = [], []
    for j in range(window, len(test)):
        x_test.append(test[j -window:j, 0])
        y_test.append(test[j, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Make predictions and inverse scale
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    train_plot = scaler.inverse_transform(train)
    
    # Calculate MSE
    MSE = mean_squared_error(y_test, predictions)

    # Plot training data, testing data, and predictions
    plt.figure(figsize=(12, 8))
    plt.plot(data.index[:len(train)], train_plot, color='blue', label='Training Data')
    plt.plot(data.index[len(train):len(train) + len(y_test)], y_test, color='green', label='Testing Data')
    plt.plot(data.index[len(train):len(train) + len(predictions)], predictions, color='red', label='Predictions')
    plt.title('LSTM Model - Training vs Testing vs Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

data = pd.read_csv("data/apple_stock_data.csv")

print("\nTesting LSTM function...")
LSTM_model(data, col="Close", window=7)
