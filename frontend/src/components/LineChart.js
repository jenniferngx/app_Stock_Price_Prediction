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

const LineChart = ({ticker}) => {
    const [chartData, setChartData] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        setLoading(true);
        setError(null);
        setChartData(null);
        const fetchData = async() => {
            try {
                // Extract training data (dates & prices)
                const response = await axios.get(
                    `http://127.0.0.1:5000/api/stock-data?ticker=${ticker}`
                );
                const {dates, prices} = response.data;

                if (!dates || dates.length === 0) {
                    throw new Error(`No data found for ticker "${ticker}". It may be delisted or unavailable.`);
                }

                setChartData({
                    labels: dates,
                    datasets: [
                        {
                            label: `${ticker} Stock Price`,
                            data: prices,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 2,
                            fill: false,
                        },
                    ],
                });
            
            } catch (error) {
                setError(`No data found for ticker "${ticker}". It may be delisted or unavailable.`);
                setChartData(null);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
        }, [ticker]);
    
    return (
        <div className='line-chart'>
            {loading ? (
                <p>Loading chart ...</p>
            ) : error ? (
                <p style={{ color: 'red', fontWeight: 'bold' }}>⚠️ {error}</p>
            ) : chartData ? (
                <Line data={chartData} />
            ) : null}     
        </div>
    );
};

export default LineChart;