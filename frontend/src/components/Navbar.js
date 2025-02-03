import React from 'react'
import './Navbar.css'
const Navbar = ({ onSearch }) => {
    return (
        <div className = "navbar">
            <div className = "logo">TradeSight</div>
            <div className = "stockSearchBox">
                <input 
                    type = "text"
                    placeholder= "Search stock ticker"
                    className = "stockSearchInput"
                    onKeyDown={(e)=> {
                        if (e.key === 'Enter'){
                            onSearch(e.target.value)
                        }
                    }}
                />
            </div>
        </div>
    );
};

export default Navbar