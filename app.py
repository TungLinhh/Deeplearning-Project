from flask import Flask, request, render_template, jsonify
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_path = "C:/sentiment_model"  # Ensure correct path
tokenizer_path = "C:/sentiment_tokenizer"

try:
    model = AlbertForSequenceClassification.from_pretrained(model_path)
    tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Sentiment labels
labels = ["very negative", "negative", "neutral", "positive", "very positive"]

def get_sentiment(text):
    try:
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        return labels[prediction]
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return "Error"

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in 'templates/'

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment_response():
    try:
        user_message = request.json.get("message", "")
        sentiment = get_sentiment(user_message)
        return jsonify({"sentiment": sentiment, "message": f"Sentiment is: {sentiment}"})
    except Exception as e:
        print(f"Error in API: {e}")
        return jsonify({"error": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)


