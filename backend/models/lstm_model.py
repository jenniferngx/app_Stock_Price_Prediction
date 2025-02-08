from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def fit_LSTM(data, col, window, lstm_unit, epoch):
    # Extract target column and scale data
    y = data[col].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    y_scaled = scaler.fit_transform(y)
    print("✅ Data normalized and scaled.")

    train_size = int(len(data) *0.8)
    train, test = y_scaled[:train_size], y_scaled[train_size - window:]
    print(f"📈 Train size: {len(train)}, Test size: {len(test)}")

    # Create training data
    x_train, y_train = [], []
    for i in range(window, len(train)):
        x_train.append(train[i-window:i, 0])
        y_train.append(train[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    print(f"✅ Training data prepared: {x_train.shape}")

    # Create testing data
    x_test, y_test = [], []
    for j in range(window, len(test)):
        x_test.append(test[j -window:j, 0])
        y_test.append(test[j, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(f"✅ Testing data prepared: {x_test.shape}")

    # Define the LSTM model
    print("🚀 Compiling LSTM model...")
    model = Sequential()
    model.add(LSTM(lstm_unit, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(130, return_sequences=False))
    model.add(Dense(100))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss='mean_squared_error')
    print("🚀 Training model...")
    model.fit(x_train, y_train, batch_size=128, epochs=epoch, verbose=0)

    # Make predictions and inverse scale
    print("✅ Model training complete. Making predictions...")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    train_plot = scaler.inverse_transform(train)
    
    # Calculate MSE
    MSE = mean_squared_error(y_test, predictions)
    print("The rmse value is:", {float(np.sqrt(MSE))})

    
    """
    # Plotting
    plt.figure(figsize = (16,8))
    plt.plot(data.index[:train_size], train_plot, label = 'train', color = 'b')
    plt.plot(data.index[train_size:], y_test.flatten(), label = 'true', color = 'r')
    plt.plot(data.index[train_size:], predictions.flatten(), label ='predict', color = 'k')
    plt.title('LSTM Model - Predictions vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("plots/lstm_model.png")
    plt.close()
    """
    return predictions.flatten()

def evaluate_LSTM(data, col, window, l, e):
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
    model.add(LSTM(l, input_shape=(x_train.shape[1], 1), return_sequences=True))
    model.add(LSTM(l, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(
        x_train, y_train, 
        validation_data = (x_test, y_test),
        batch_size=32, 
        epochs=e, 
        verbose=0)
    validation_loss = history.history['val_loss'][-1]
    return validation_loss

def find_params_LSTM(data, col, samples=20, windows=[7,14,21,30], L=[50,100,150], E=[50,100]):
    best_validation_loss = float('inf')
    best_params = None
    
    for _ in range(samples):
        window = random.choice(windows)
        lstm_unit = random.choice(L)
        epoch = random.choice(E)

        print(f"\nTesting: Window={window}, LSTM Units={lstm_unit}, Epochs={epoch}")
        try:
            # Evaluate LSTM with current parameter set
            validation_loss = evaluate_LSTM(data, col, window, lstm_unit, epoch)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_params = (window, lstm_unit, epoch)
                print(f"New best validation loss: {best_validation_loss:.4f}")
        except ValueError as e:
            # Handle test set size issues or other errors
            print(f"Skipping combination Window={window}, LSTM Units={lstm_unit}, Epochs={epoch} due to error: {e}")
            continue
    print(f"Best LSTM parameters: Windows={best_params[0]}, LSTM Units={best_params[1]}, Epochs={best_params[2]} with validation loss: {best_validation_loss:.4f}")   
    return best_params

# Testing functionality
'''
data = pd.read_csv("data/AAPL_20210101_20211231.csv")

print("\nTesting fihn_params_LSTM function...")
best_params = find_params_LSTM(data, col="Close") #result: window=7, lstm=150, epoch=50

print("\nTesting LSTM function...")
fit_LSTM(data, col="Close", window=7, lstm_unit=150, epoch=50)
'''
