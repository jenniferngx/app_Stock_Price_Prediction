import React, { useEffect, useState } from 'react';
import axios from 'axios'
const PredictionForm = ({ticker, onSubmit}) => {
    const [startDate, setStartDate] = useState('');
    const [daysAhead, setDaysAhead] = useState(7);
    const [model, setModel] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        const endDate = new Date().toISOString().split('T')[0]
        const predictionParams = {ticker, start_date: startDate, end_date: endDate, model, days_ahead: daysAhead };

        try {
            const response = await axios.post(`http://127.0.0.1:5000/predict`, predictionParams);
            console.log('Raw Predictions received:', response.data.predictions);
            if (response.data.predictions){
                const formattedPredictions = response.data.predictions.map((price, index) => ({
                    date: new Date(Date.now() + (index * 24 * 60 * 60 * 1000)).toISOString().split('T')[0],
                    price: parseFloat(price)  // Ensure it's a number
                }));
                console.log("Formatted Predictions:", formattedPredictions);
                onSubmit(formattedPredictions);
            } else {
                console.log("No predictions received");
            }
        } catch (error) {
            console.error('Error fetching predictions:', error);
        }
    };

    return (
        <form onSubmit={handleSubmit} className='prediction-form'>
            <div className='start-date'>
                <label htmlFor='start-date'>Start Date: </label>
                <input
                    type = "date"
                    id="start-date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    required
                />
            </div>
            <div className='model'>
                <label htmlFor='model'>Model: </label>
                <select
                    id="model"
                    value={model}
                    onChange={(e)=>setModel(e.target.value)}
                    required
                >
                    <option value="" disabled hidden> Select a model for prediction</option>
                    <option value="arima"> ARIMA </option>
                    <option value="lstm"> LSTM </option>
                    <option value="best"> Best-performing Model: Let us choose for you!</option>
                </select>
            </div>
            <button type="submit">Submit</button>
        </form>
    );
};

export default PredictionForm;