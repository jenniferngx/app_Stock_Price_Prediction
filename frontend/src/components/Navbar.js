import React from 'react'
import './Navbar.css'
import {Link, useNavigate} from 'react-router-dom'

const Navbar = ({ onSearch }) => {
    const navigate = useNavigate();
    
    return (
        <div className = "navbar">
            <Link to ="/" className = "logo">TradeSight</Link>
            <div className = "stockSearchBox">
                <input type = "text" className = "stockSearchInput" 
                    placeholder= "Search stock ticker"
                    onKeyDown={(e)=> {
                        if (e.key === 'Enter'){
                            const ticker = e.target.value.trim()
                            onSearch(ticker);
                            navigate(`/${ticker}`);
                        }
                    }}
                />
            </div>
        </div>
    );
};

export default Navbar