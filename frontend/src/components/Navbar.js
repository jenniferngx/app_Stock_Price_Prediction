import React, {useState} from 'react'
import './Navbar.css'
import {Link, useNavigate} from 'react-router-dom'

const Navbar = () => {
    const [ticker, setTicker] = useState('');
    const navigate = useNavigate();

    const handleSearch = (e) => {
        e.preventDefault();
        if (ticker.trim()) {
            navigate(`/${ticker}`);
        }
    };
    
    return (
        <div className = "navbar">
            <Link to ="/" className = "logo">TradeSight</Link>
            <form className = "stockSearchBox" onSubmit={handleSearch}>
                <input 
                    type = "text" 
                    className = "stockSearchInput" 
                    placeholder= "Enter stock ticker"
                    value = {ticker}
                    onChange ={(e)=> setTicker(e.target.value.trim())}
                />
            </form>
        </div>
    );
};

export default Navbar;