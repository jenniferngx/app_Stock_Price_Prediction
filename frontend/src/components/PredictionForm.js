import React, { useEffect, useState } from 'react';

const PredictionForm = ({ticker, onSubmit}) => {
    const [startDate, setStartDate] = useState('');
    const [model, setModel] = useState('best');

    const handleSubmit = (e) => {
        e.preventDefault();
        const endDate = new Date().toISOString().split('T')[0]
        onSubmit({ticker, startDate, endDate, model});
    };

    return (
        <form onSubmit={handleSubmit} className='prediction-form'>
            <div>
                <label htmlFor='start-date'>Start Date:</label>
            </div>
            <div>
                <label htmlFor='model'>Model:</label>
            </div>
            <button type="submit">Submit</button>
        </form>
    );
};

export default PredictionForm;