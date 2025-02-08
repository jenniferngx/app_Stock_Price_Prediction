import React from "react";

const PredictionResult = ({predictions = []}) => {
    console.log("🟢 Rendering PredictionResult with predictions:", predictions);

    if (!predictions.length) {
        console.log("⚠️ No predictions, table not rendering.");
        return null;
    }
    return (
        <div className = "prediction-table">
            <h3> Predicted Prices</h3>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Predicted Price</th>
                    </tr>
                </thead>
                <tbody>
                    {predictions.map((item, index) => (
                        <tr key={index}>
                            <td>{item.date}</td>
                            <td>${item.price.toFixed(2)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default PredictionResult;