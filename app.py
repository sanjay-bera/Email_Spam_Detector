from flask import Flask, render_template, request, jsonify
import pickle
import string
import os

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorization.pkl", "rb"))

def transform_text(text):
    """
    Minimal, production-safe preprocessing.
    TF-IDF handles tokenization internally.
    """
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        message = request.form.get("message", "").strip()

        if not message:
            return jsonify({"error": "Please enter a message"})

        transformed_message = transform_text(message)

        message_vector = vectorizer.transform([transformed_message])

        prediction = model.predict(message_vector)[0]

        result = "SPAM" if prediction == 1 else "NOT SPAM"

        return jsonify({
            "result": result,
            "message": message
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
