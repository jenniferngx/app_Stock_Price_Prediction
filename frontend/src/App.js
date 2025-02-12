import './App.css';
import React, {useState} from 'react';
import { BrowserRouter as Router, Route, Routes} from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import StockDetailsPage from './pages/StockDetailsPage.js';
import PredictionPage from './pages/PredictionPage';

const App = () => {
  return (
    <Router>
      <Navbar/>
      <Routes>
        <Route path='/' element={<HomePage/>} />
        <Route path='/:ticker' element={<StockDetailsPage/>} />
        <Route path='/:ticker/predictions' element={<PredictionPage/>} />
      </Routes>
    </Router>
  );
};

export default App;