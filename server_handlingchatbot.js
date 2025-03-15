const express = require('express');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');
const app = express();
const port = 3000;

app.use(bodyParser.json());

app.post('/predict', (req, res) => {
  const userInput = req.body;

  const inputData = JSON.stringify(userInput);

  PythonShell.run('predict.py', { args: [inputData] }, (err, result) => {
    if (err) {
      return res.status(500).json({ error: "Error in model prediction" });
    }

    const predictionResult = JSON.parse(result[0]);

    res.json(predictionResult);
  });
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
