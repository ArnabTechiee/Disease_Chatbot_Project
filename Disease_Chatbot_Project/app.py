from flask import Flask, render_template, request, jsonify
from chatbot.chatbot import chatbot_response, process_input  # Import chatbot functions

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

user_data = {}  # Store user responses
step = 0  # Track conversation step

@app.route("/")
def home():
    return render_template("index.html")  # Ensure this file exists in "frontend/templates/"

@app.route("/get_response", methods=["POST"])
def get_response():
    global step, user_data

    user_message = request.json.get("message", "").strip()

    if step == 0:
        user_data = {}  # Reset user data

    bot_response = chatbot_response(user_message, step, user_data)

    if step < 9:
        step += 1
    elif step == 9:
        step = 10  # Move to restart/exit prompt
    elif step == 10:
        if process_input(user_message) == "Yes":
            step = 0  # Restart diagnosis
        else:
            step = 0  # Ensure chatbot doesn't break after exit

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
