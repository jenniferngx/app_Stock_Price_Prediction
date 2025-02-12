import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js'

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const PredictionChart = ({ticker, predictions, startDate}) => {
    console.log('Fetching training data with:', { ticker, startDate });
    const [chartData, setChartData] = useState(null);

    useEffect(() => {
        const fetchData = async() => {
            try {
                // Extract training data (dates & prices)
                const response = await axios.get(
                    `http://127.0.0.1:5000/api/train-data`, {
                        params: {
                            ticker: ticker,
                            start_date: startDate,
                        },
                    }
                );
                const {dates, prices} = response.data;

                // Extract predicted data (dates & prices)
                const predictedDates = predictions.map((p) => p.date);
                const predictedPrices = predictions.map((p) => p.price);

                setChartData({
                    labels: [...dates, ...predictedDates],
                    datasets: [
                        {
                            label: `${ticker} Stock Price`,
                            data: prices,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 2,
                            fill: false,
                        },
                        {
                            label: `${ticker} Predictions`,
                            data: [...new Array(dates.length).fill(null), ...predictedPrices], // Append predictions
                            borderColor: 'rgba(255, 99, 132, 1)', // Red for predicted data
                            borderWidth: 2,
                            borderDash: [5,5],
                            fill: false,
                        },
                    ],
                });
            } catch (error) {
                console.error('Error fetching stock data:', error);
            }
        };
        fetchData();
        }, [ticker, predictions, startDate]);
    
    return (
        <div className='line-chart'>
            {chartData ? <Line data = {chartData}/>: <p>Loading chart ...</p>}
        </div>
    );
};

export default PredictionChart;