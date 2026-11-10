"""Flask web app for the emotion-aware chatbot."""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from emotion import EmotionClassifier

BASE_DIR = Path(__file__).resolve().parent
RESPONSES_PATH = BASE_DIR / "responses.json"
MAX_MESSAGE_LENGTH = 1000
DEFAULT_RESPONSE = "I am not sure how to respond to that, but I am here to listen."

app = Flask(__name__)
classifier = EmotionClassifier()

with open(RESPONSES_PATH, encoding="utf-8") as fh:
    RESPONSES: dict[str, str] = json.load(fh)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def get_bot_response():
    """Accepts form data or JSON with a `msg` field."""
    payload = request.get_json(silent=True) or request.form
    message = str(payload.get("msg", "")).strip()

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."}), 400

    result = classifier.classify(message)
    result["response"] = RESPONSES.get(result["emotion"], DEFAULT_RESPONSE)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": classifier.has_model})


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        port=int(os.environ.get("PORT", 5001)),
    )
