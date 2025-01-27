import os
from flask import Flask, render_template, request, jsonify
from app.data_collection import fetch_stock_data
from app.model import predict_stock_price

print("Template search path:", os.path.join(os.getcwd(), "templates"))
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form.get("ticker")
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    try:
        data = fetch_stock_data(ticker, start_date, end_date)
        prediction = predict_stock_price(data)
        return jsonify({"ticker": ticker, "prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
