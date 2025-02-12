from flask import Flask, request, jsonify
import pandas as pd
from models.lstm_model import forecast_LSTM  
from models.arima_model import fit_ARIMA 
from data_utils import fetch_stock_data, store_stock_data
from datetime import datetime, timedelta
app = Flask(__name__)
from flask_cors import CORS  # To communicate with React frontend
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/api/stock-data', methods=['GET'])
def get_stock_data_3Y():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker is required'}), 400    
    
    # Calculate last 3 years' date range 
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    # Fetch data for last 3 years
    data = fetch_stock_data(ticker, start_date)
    filepath = store_stock_data(ticker, data, start_date)
    if data.empty:
        return jsonify({'error': 'No data found for the given ticker and dates'}), 404
    
    return jsonify({
        'dates': data['Date'].astype(str).tolist(),
        'prices': data['Close'].tolist()
    })

@app.route('/api/train-data', methods=['GET'])
def get_training_data():
    ticker = request.args.get('ticker')
    start_date = request.args.get('start_date')

    if not ticker or not start_date:
        return jsonify({'error': 'Ticker and start_date are required'}), 400

    try:
        train_data = fetch_stock_data(ticker, start_date)
        if train_data.empty:
            print(f"No data found for Ticker: {ticker} from {start_date}")
            return jsonify({'error': 'No data found for the given ticker and dates'}), 404

        return jsonify({
            'dates': train_data['Date'].astype(str).tolist(),
            'prices': train_data['Close'].tolist()
        })
    except Exception as e:
        print(f"Error fetching training data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from frontend
    print("Received a POST request to /predict")
    data = request.json  
    ticker = data.get('ticker')
    start_date = data.get('start_date')
    model = data.get('model')
    days_ahead = int(data.get('days_ahead'))
    print(f"🎯 Prediction request details: Ticker={ticker}, Start={start_date}, Model={model}, Days Ahead={days_ahead}")
    
    try:
        # Fetch data 
        print(f"Fetching stock data according to user-specified time range")
        stock_data = fetch_stock_data(ticker, start_date)

        if stock_data is None or stock_data.empty:
            return jsonify({'error': f'No data found for ticker {ticker}'})
        print(f"✅ Stock data loaded: {stock_data.shape[0]} rows")

        if model == 'lstm':
            print("🚀 Running LSTM model...")
            predictions = forecast_LSTM(stock_data, "Close", days_ahead)
        elif model == 'arima':
            predictions = fit_ARIMA(stock_data, "Close", days_ahead)
        else:
            return jsonify({'error': 'Invalid model selected'}), 400
        print(f"✅ Returning predictions: {predictions.tolist()[:5]}")
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        print(f" ERROR: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
