import React from 'react';
import ReactMarkdown from 'react-markdown';
import './ChatMessage.css';

const ChatMessage = ({ message, isUser }) => (
  <div className={`chat-message ${isUser ? 'user' : 'bot'}`}>
    <div className="message-content">
      <ReactMarkdown>{message}</ReactMarkdown>
    </div>
  </div>
);

export default ChatMessage;