import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

// Connect to the backend server
const socket = io('http://localhost:3000');

const App = () => {
  const [questions, setQuestions] = useState([
    "What is your age?",
    "What is your gender? (Male/Female)",
    "Do you have any allergies? (Yes/No)",
    "Do you have a history of heart disease?",
    "Do you smoke?",
    "How often do you exercise?",
    "Do you have any chronic conditions? (e.g. diabetes)",
    "Are you currently on any medication?",
    "Do you have any family history of serious illnesses?",
    "Have you had any recent surgeries?",
    "Are you experiencing any pain or discomfort?",
    "Have you traveled outside the country recently?"
  ]);

  const [answers, setAnswers] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = () => {
    if (userInput) {
      // Add user input to chat history
      setChatHistory([...chatHistory, { sender: 'user', message: userInput }]);
      setAnswers([...answers, userInput]);
      setUserInput('');
      
      // Move to the next question or finish
      if (currentQuestionIndex < questions.length - 1) {
        setCurrentQuestionIndex(currentQuestionIndex + 1);
      } else {
        socket.emit('userAnswers', answers.concat(userInput));
        setIsLoading(true);

        // Show a bot message that it's processing
        setChatHistory([
          ...chatHistory,
          { sender: 'bot', message: 'Analyzing your responses and generating further questions...' }
        ]);
      }
    }
  };

  useEffect(() => {
    socket.on('generatedQuestions', (data) => {
      console.log('Generated Questions:', data);
      setIsLoading(false);
      setChatHistory([
        ...chatHistory,
        { sender: 'bot', message: "Here are the additional questions based on your responses:" },
        { sender: 'bot', message: data.questions }
      ]);
    });

    socket.on('confirmation', (data) => {
      console.log('Backend confirmation received:', data);
    });

  }, [chatHistory]);

  return (
    <div style={styles.container}>
      {/* Top Bar */}
      <div style={styles.topBar}>
        <div style={styles.topBarContent}>
          <button style={styles.topBarButton}>Settings</button>
          <button style={styles.topBarButton}>Profile</button>
        </div>
      </div>

      {/* Chat Area */}
      <div style={styles.chatArea}>
        <div style={styles.chatHistory}>
          {/* Displaying the chat history */}
          {chatHistory.map((chat, index) => (
            <div key={index} style={styles.messageContainer(chat.sender)}>
              <div style={styles.messageBox(chat.sender)}>
                {chat.message}
              </div>
            </div>
          ))}

          {/* Show "Analyzing" message when waiting for the backend to respond */}
          {isLoading && (
            <div style={styles.loadingMessage}>
              <span>Analyzing your responses...</span>
            </div>
          )}

          {/* Show the current question */}
          {!isLoading && currentQuestionIndex < questions.length && (
            <div style={styles.messageContainer('bot')}>
              <div style={styles.messageBox('bot')}>
                {questions[currentQuestionIndex]}
              </div>
            </div>
          )}
        </div>

        <div style={styles.inputContainer}>
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            style={styles.inputField}
            disabled={isLoading}
            placeholder="Type your answer..."
          />
          <button
            onClick={handleSubmit}
            style={styles.submitButton}
            disabled={isLoading || !userInput.trim()}
          >
            {isLoading ? 'Processing...' : 'Submit'}
          </button>
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: '#111', // Dark background
    color: '#fff',
    fontFamily: '"Roboto", sans-serif',
  },
  topBar: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1A1A1A',
    padding: '15px 0',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.5)',
  },
  topBarContent: {
    display: 'flex',
    justifyContent: 'flex-end',
    width: '100%',
    padding: '0 40px',
  },
  topBarButton: {
    backgroundColor: '#4CAF50',
    color: '#fff',
    border: 'none',
    padding: '12px 25px',
    marginLeft: '20px',
    borderRadius: '25px',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  topBarButtonHover: {
    backgroundColor: '#45a049',
  },
  chatArea: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    padding: '30px 50px',
    overflowY: 'auto',
    backgroundColor: 'rgba(30, 30, 30, 0.8)', // Chat area with transparency
    backdropFilter: 'blur(15px)', // Background blur effect
    boxShadow: 'inset 0 0 10px rgba(0, 0, 0, 0.5)',
  },
  chatHistory: {
    flex: 1,
    marginBottom: '30px',
    overflowY: 'auto',
  },
  messageContainer: (sender) => ({
    margin: '10px 0',
    textAlign: sender === 'user' ? 'right' : 'left',
  }),
  messageBox: (sender) => ({
    display: 'inline-block',
    padding: '12px 18px',
    backgroundColor: sender === 'user' ? '#4CAF50' : '#444',
    borderRadius: '20px',
    maxWidth: '70%',
    wordWrap: 'break-word',
    fontSize: '16px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
  }),
  loadingMessage: {
    margin: '20px 0',
    textAlign: 'center',
    fontSize: '18px',
    color: '#bbb',
  },
  inputContainer: {
    padding: '10px',
    backgroundColor: '#333',
    borderRadius: '20px',
    boxShadow: '0px -2px 10px rgba(0, 0, 0, 0.5)',
    marginBottom: '20px',
  },
  inputField: {
    width: '100%',
    padding: '14px 20px',
    borderRadius: '20px',
    border: 'none',
    marginBottom: '10px',
    backgroundColor: '#444',
    color: '#fff',
    fontSize: '16px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
  },
  submitButton: {
    width: '100%',
    padding: '14px',
    backgroundColor: '#4CAF50',
    color: '#fff',
    border: 'none',
    borderRadius: '20px',
    fontSize: '18px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
  },
  submitButtonHover: {
    backgroundColor: '#45a049',
  },
};

export default App;
