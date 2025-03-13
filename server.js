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
  socket.on('allAnswersReceived', (answerMap) => {
    console.log('Final answers map:', answerMap); 
  });

  socket.on('disconnect', () => {
    console.log('A user disconnected');
  });
});

server.listen(3000, () => {
  console.log('Server is running on http://localhost:3000');
});
