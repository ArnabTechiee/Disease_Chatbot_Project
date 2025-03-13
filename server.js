import express from 'express';
import http from 'http';
import { Server as SocketServer } from 'socket.io';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';
const app = express();
const server = http.createServer(app);
const io = new SocketServer(server, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"]
  }
});

app.use(cors({
  origin: "http://localhost:5173",
  methods: ["GET", "POST"]
}));

let userAnswers = [];

const genAI = new GoogleGenerativeAI("");
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

io.on('connection', (socket) => {
  console.log('A user connected');

  socket.on('userAnswers', async (answers) => {
    userAnswers = answers;
    console.log('Received answers:', userAnswers);

    try {
      // Step 1: Use answers to generate a prompt for the Gemini API
      const prompt = `Analyze the following answers and generate further questions for the user based on their health responses: ${userAnswers.join(', ')}`;
      
      // Step 2: Send the prompt to Gemini API for question generation
      const result = await model.generateContent(prompt);
      const newQuestions = result.response.text(); // This should be a list of generated questions

      console.log('Generated questions:', newQuestions);

      // Step 3: Send the new questions back to the frontend
      socket.emit('generatedQuestions', { questions: newQuestions });
    } catch (error) {
      console.error('Error generating questions:', error);
      socket.emit('generatedQuestions', { questions: ['Sorry, there was an error generating additional questions.'] });
    }
  });

  socket.on('allAnswersReceived', (answerMap) => {
    console.log('Final answers map:', answerMap);  // Logging the question-answer map to console
  });

  socket.on('disconnect', () => {
    console.log('A user disconnected');
  });
});

server.listen(3000, () => {
  console.log('Server is running on http://localhost:3000');
});
