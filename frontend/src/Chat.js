import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './Chat.css';

function Chat() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! Before we start, can you tell me are you a student or teacher?' }
  ]);
  const [input, setInput] = useState('');
  const navigate = useNavigate();

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessage = { role: 'user', content: input };
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    setInput('');
    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: updatedMessages }),
      });
      if (!response.ok) {
        console.error("Chat API error", response.statusText);
        return;
      }
      const data = await response.json();
      const assistantReply = data.response;
      const newMessages = [...updatedMessages, { role: 'assistant', content: assistantReply }];
      setMessages(newMessages);

      if (assistantReply.includes("Found a match")) {
        const conversationText = newMessages.map(msg => `${msg.role}: ${msg.content}`).join("\n");
        const summaryResponse = await fetch('http://127.0.0.1:8000/api/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ conversation: conversationText })
        });
        if (!summaryResponse.ok) {
          console.error("Summarize API error", summaryResponse.statusText);
          return;
        }
        const summaryData = await summaryResponse.json();
        localStorage.setItem('searchQuery', summaryData.search_query);
        navigate('/matches');
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-container">
      <motion.div
        className="chat-box"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="chat-title">Chat with EduConnect.AI</h2>
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              className={`chat-message ${msg.role}`}
              initial={{ opacity: 0, x: msg.role === 'assistant' ? -50 : 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
            >
              <strong>{msg.role === 'assistant' ? 'Assistant' : 'You'}:</strong> {msg.content}
            </motion.div>
          ))}
        </div>
        <div className="chat-input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="chat-input"
          />
          <button onClick={sendMessage} className="chat-send-button">Send</button>
        </div>
      </motion.div>
    </div>
  );
}

export default Chat;
