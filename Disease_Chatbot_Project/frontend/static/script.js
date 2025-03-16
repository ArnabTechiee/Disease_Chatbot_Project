document.addEventListener("DOMContentLoaded", function () {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    function appendMessage(sender, message, isLoading = false) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add(sender === "user" ? "user-message" : "bot-message");

        if (isLoading) {
            messageDiv.innerHTML = `<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span>`;
        } else {
            messageDiv.innerHTML = sender === "user" ? message : `<span class="bot-icon">🤖</span> ${message}`;
        }

        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return messageDiv;
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        appendMessage("user", message);
        userInput.value = "";

        const loadingMessage = appendMessage("bot", "", true);

        fetch("/get_response", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message }),
        })
        .then(response => response.json())
        .then(data => {
            chatBox.removeChild(loadingMessage);
            appendMessage("bot", data.response);
        })
        .catch(error => {
            console.error("Error:", error);
            chatBox.removeChild(loadingMessage);
            appendMessage("bot", "⚠ Sorry, something went wrong. Please try again.");
        });
    }

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") sendMessage();
    });
});
