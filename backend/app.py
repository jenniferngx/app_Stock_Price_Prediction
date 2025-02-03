from flask import Flask, request, jsonify
from flask_cors import CORS  # To enable communication with the React frontend
import pandas as pd
from models.lstm_model import fit_LSTM  # Example of importing your LSTM model
from data_utils import fetch_stock_data
from datetime import datetime, timedelta
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/api/stock-data', methods=['GET'])
def get_stock_data_3Y():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker is required'}), 400    
    
    # Calculate the last 3 years' date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    # Fetch data for the last 3 years' date range
    data, filepath = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        return jsonify({'error': 'No data found for the given ticker and dates'}), 404
    
    return jsonify({
        'dates': data['Date'].astype(str).tolist(),
        'prices': data['Close'].tolist()
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from frontend
    ticker = data.get('ticker')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    days_ahead = int(data.get('days_ahead'))

    # Load and process data (example using your model)
    try:
        # Fetch data and preprocess (adjust as per your function structure)
        stock_data = pd.read_csv(f"data/{ticker}_{start_date}_{end_date}.csv")
        predictions = fit_LSTM(stock_data, "Close", 7, 150, 50)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
