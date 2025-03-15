import React, { useState, useEffect } from 'react';
import { Paperclip, Send } from 'lucide-react';
import ChatMessage from '../ChatMessage/ChatMessage';
import './ChatInterface.css';
import rhPanel from '../../rh_panel.png'; // Import the image

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [conversationId, setConversationId] = useState(null);

  // Start a new chat when component mounts
  useEffect(() => {
    const hasStarted = sessionStorage.getItem('chatStarted');
    
    const startChat = async () => {
      try {
        const response = await fetch('http://localhost:5000/chat/start');
        const data = await response.json();
        
        // Store the conversation ID
        setConversationId(data.conversation_id);
        
        // Add the greeting message
        setMessages([{ text: data.response, isUser: false }]);
      } catch (error) {
        console.error('Failed to start chat:', error);
      }
    };
    
    startChat();
  }, []);

  const handleSend = async (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      
      // Add user message to UI immediately
      setMessages([...messages, { text: inputValue, isUser: true }]);
      
      try {
        const response = await fetch('http://localhost:5000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            message: inputValue,
            conversation_id: conversationId
          }),
        });
        
        const data = await response.json();
        
        // Add response to messages
        setMessages(prev => [...prev, { text: data.response, isUser: false }]);
      } catch (error) {
        console.error('Error:', error);
      }
  
      setInputValue('');
    }
  };

  return (
    <div className="chat-layout">
      <div className="chat-container">
        <div className="chat-panel">
          <div className="chat-header">
            <span className="chat-title">Nectar 360 Audience Builder</span>
          </div>
          <div className="messages-container">
            {messages.map((msg, idx) => (
              <ChatMessage key={idx} message={msg.text} isUser={msg.isUser} />
            ))}
          </div>
        </div>

        <div className="input-container">
          <form onSubmit={handleSend} className="input-form">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask Nectar ..."
              className="message-input"
            />
            <div className="action-buttons">
              <button type="button" className="action-button">
                <Paperclip className="icon" color="#6B7280" />
              </button>
              <button type="submit" className="action-button">
                <Send className="icon" color="white" />
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* <div className="ui-container">
        <img src={rhPanel} alt="Right panel" className="panel-image" />
      </div> */}
    </div>
  );
};

export default ChatInterface;