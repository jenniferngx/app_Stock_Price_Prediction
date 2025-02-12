import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import PredictionResult from '../components/PredictionResult';
import PredictionChart from '../components/PredictionChart';
import './PredictionPage.css';

const PredictionPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { ticker, predictions, start_date} = location.state || {};
  console.log("Start date in Prediction Page:", start_date)
  
  if (!predictions) { // Redirect to home if no predictions available
    navigate(`/${ticker}`);
    return null;
  }

  return (
    <div className="prediction-page">

      {/* LineChart Component */}
      <div className="chart-container">
        <h1>Prediction Results for {ticker}</h1>
        <PredictionChart 
          ticker={ticker} 
          predictions={predictions} 
          startDate={start_date}/>
      </div>

      {/* PredictionResult Component */}
      <div className = "table-container">
        <PredictionResult predictions={predictions} />
        <button onClick={() => navigate(`/`)}>Go Home</button>
      </div>

    </div>
  );
};

export default PredictionPage;
