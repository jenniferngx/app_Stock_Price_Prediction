import React, {useState} from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import LineChart from '../components/LineChart';
import PredictionForm from '../components/PredictionForm';
import './StockDetailsPage.css';

const StockDetailsPage = () => {
  const { ticker } = useParams();
  const navigate = useNavigate();

  const handleSubmit = (predictions, startDate) => {
    navigate(`/${ticker}/predictions`, { state: { ticker, predictions, start_date:startDate} });
  };

  return (
    <div className="stock-details-page">
        <div className='chart-container'>
            <LineChart ticker={ticker}/>
        </div> 
        <div className='form-container'>
            <PredictionForm ticker={ticker} onSubmit={handleSubmit}/>
        </div>
    </div>
  );
};

export default StockDetailsPage;
