import React, { useState, useEffect, useRef } from 'react';
import './Chat.css';
import logo from './logo.png';
import sendIcon from './send.png';

function Chat() {
    const [input, setInput] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const chatEndRef = useRef(null);

    const handleInputChange = (event) => {
        setInput(event.target.value);
    };

    const handleSend = async () => {
        if (input.trim() === '') return;

        setIsProcessing(true);
        setChatHistory([...chatHistory, { type: 'user', message: input }]);

        const response = await fetch('http://localhost:8001/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: input }),
        });

        if (response.ok) {
            const data = await response.json();
            setChatHistory((prevHistory) => [...prevHistory, { type: 'bot', message: data.response }]);
        } else {
            setChatHistory((prevHistory) => [...prevHistory, { type: 'bot', message: 'Failed to get response' }]);
        }

        setInput('');
        setIsProcessing(false);
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            handleSend();
        }
    };

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatHistory, isProcessing]);

    return (
        <div className="chat-container">
            <div className="sidebar">
                <img src={logo} alt="Hyperledger Logo" className="logo" />
                <div className="today-section">
                    <div className="today-title">Today</div>
                    <div className="greeting">What is Hyperleger?</div>
                    <div className="greeting">Setting up hyperledger iroha</div>
                    <div className="greeting">What are the current good first is..</div>
                    
                    <div className="today-title">Yesterday</div>
                    <div className="greeting">What principle does the Iroha API.. </div>
                    <div className="greeting">How are permissions in iroha cate..</div>
                    <div className="greeting">What kind of information can be ..</div>
                    <div className="greeting">What is the purpose of a comman..</div>

                    <div className="today-title">Few days back</div>
                    <div className="greeting">Give an example of an entity that ..</div>
                    <div className="greeting">Tell me the tech stack of iroha.</div>
                </div>
            </div>
            <div className="chat-section">
                <div className="chat-history">
                    {chatHistory.map((item, index) => (
                        <div key={index} className={`chat-message ${item.type}`}>
                            <span className="message-text">{item.message}</span>
                        </div>
                    ))}
                    <div ref={chatEndRef} />
                </div>
                <div className="input-container">
                    <input
                        className="chat-input"
                        value={input}
                        onChange={handleInputChange}
                        onKeyPress={handleKeyPress}
                        placeholder="Enter your Query"
                        disabled={isProcessing}
                    />
                    <button className="chat-button" onClick={handleSend} disabled={isProcessing}>
                        <img src={sendIcon} alt="Send" className="send-icon" />
                    </button>
                </div>
            </div>
        </div>
    );
}

export default Chat;