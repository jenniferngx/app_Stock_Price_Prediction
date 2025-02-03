import './App.css';
import React, {useState} from 'react'
import Navbar from './components/Navbar';
import LineChart from './components/LineChart';
import PredictionForm from './components/PredictionForm';

function App() {
  const [ticker, setTicker] = useState('');
  const [predictionParams, setPredictionParams] = useState(null);

  const handleSearch = (ticker) => {
    setTicker(ticker)
  };

  const handlePredictionSubmit = (params) => {
    setPredictionParams(params);
  };

  return (
    <div>
      <Navbar onSearch={handleSearch}/>
      {ticker ?(
        <div className='main-container'>
          <div className='chart-container'>
            <LineChart ticker={ticker}/>
          </div> 
          <div className='form-container'>
            <PredictionForm ticker={ticker} onSubmit={handlePredictionSubmit}/>
          </div>
        </div>
      ): (
        <div className = "welcome">
        <h1>Welcome to TradeSight</h1>
        <p>Start by searching for a stock ticker in the search box.</p>
      </div>
      )}
    </div>
  );
}

export default App;