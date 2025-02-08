import './App.css';
import React, {useState} from 'react';
import { BrowserRouter as Router, Route, Routes, useParams } from 'react-router-dom';
import Navbar from './components/Navbar';
import LineChart from './components/LineChart';
import PredictionForm from './components/PredictionForm';
import PredictionResult from './components/PredictionResult'
import Home from './pages/HomePage';

const StockDetails = ({onSubmit, predictions}) => {
  let {ticker} = useParams();

  return (
    <div className='main-container'>
      <div className='chart-container'>
        <LineChart ticker={ticker} prediction={predictions}/>
      </div> 
      <div className='form-container'>
        <PredictionForm ticker={ticker} onSubmit={onSubmit}/>
        <PredictionResult predictions = {predictions}/>
      </div>
    </div>
  );
};

function App() {
  const [ticker, setTicker] = useState('');
  const [predictions, setPredictions] = useState([]);

  const handleSearch = (ticker) => {setTicker(ticker)};

  const handlePredictionSubmit = (predictedData) => {
    console.log("Updating predictions in state:", predictedData);
    setPredictions(predictedData);
  };

  return (
    <Router>
      <Navbar onSearch={handleSearch}/>
      <Routes>
        <Route path='/' element={<Home onSearch={handleSearch}/>}/>
        <Route path='/:ticker' element={<StockDetails onSubmit={handlePredictionSubmit} predictions={predictions} />} />
      </Routes>
    </Router>
  );
}

export default App;