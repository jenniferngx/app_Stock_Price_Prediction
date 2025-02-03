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

    useEffect(() => {
        const fetchData = async() => {
            try {
                const response = await axios.get(
                    `http://127.0.0.1:5000/api/stock-data?ticker=${ticker}`
                );
                const {dates, prices} = response.data;
                setChartData({
                    labels: dates,
                    datasets: [
                        {
                            labels: `${ticker} Stock Price`,
                            data: prices,
                            borderWidth: 2,
                            fill: false,
                        },
                    ],
                });
            } catch (error) {
                console.error('Error fetching stock data:', error);
            }
        };
        fetchData();
        }, [ticker]);
    
    return (
        <div className='line-chart'>
            {chartData ? <Line data = {chartData}/>: <p>Loading chart ...</p>}
        </div>
    );
};

export default LineChart;