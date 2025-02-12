import React, { useState } from 'react';
import axios from 'axios'
const PredictionForm = ({ticker, onSubmit}) => {
    const [startDate, setStartDate] = useState('');
    const [daysAhead, setDaysAhead] = useState(7);
    const [model, setModel] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);

        const predictionParams = {ticker, start_date: startDate, model, days_ahead: daysAhead };
        console.log('Start date in PredictionForm:', startDate);
        
        try {
            setIsLoading(true);
            const response = await axios.post(`http://127.0.0.1:5000/predict`, predictionParams);
            
            if (response.data.predictions){
                const formattedPredictions = response.data.predictions.map((price, index) => ({
                    date: new Date(Date.now() + (index * 24 * 60 * 60 * 1000)).toISOString().split('T')[0],
                    price: parseFloat(price) 
                }));
                onSubmit(formattedPredictions, startDate);
            } else {
                setError("Unexpected error: No predictions received.");

            }
        } catch (error) {
            if (error.response && error.response.data.error) {
                setError(error.response.data.error);  // Display API error message
            } else {
                setError("An unexpected error occurred.");
            }
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className='prediction-form-container'>
            {isLoading? (
                <div className='loading-container'> 
                    <div className='spinner'></div>
                    <p>Please wait while we generate our predictions...</p>
                </div>
            ): (
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
                <div className='days-ahead'>
                    <label htmlFor='days-ahead'>Number of days to predict: </label>
                    <input
                        type = "number"
                        id="days-ahead"
                        value={daysAhead}
                        onChange={(e) => setDaysAhead(e.target.value)}
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
            )}
        </div>
    );
};

export default PredictionForm;