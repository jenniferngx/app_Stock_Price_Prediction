import React from 'react'
import {motion} from "framer-motion";
import './HomePage.css';

const HomePage = () => {
    return (
        <div className='homepage-container'>
            <img className='background-gif'
                    alt = "Homepage GIF" 
                    src = "/background.gif"
                    />
            <motion.div 
                className='text-overlay'
                initial={{opacity:0, y:-20}}
                animate={{opacity:1, y:0}}
                transition={{duration:1}}>
                <p> AI-powered stock predictions at your fingertips </p>
                <h1> 
                    <span>Get started</span>
                    <br />
                    <span>by searching</span>
                    <br />
                    <span>a stock ticker</span>
                </h1>
                <p> Disclaimer: This is not financial advice.</p>
            </motion.div>
        </div>

    );
};

export default HomePage;