import argparse
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


# Function to load the tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {tokenizer_path}")
    return tokenizer


# Function to load the embedding matrix (if applicable)
def load_embedding_matrix(embedding_matrix_path):
    embedding_matrix = np.load(embedding_matrix_path)
    print(f"Embedding matrix loaded from {embedding_matrix_path}")
    return embedding_matrix


# Function to load the model
def load_model(model_path, model_type):
    if model_type in ['cnn', 'lstm']:
        # Load a Keras model (e.g., CNN or LSTM)
        model = tf.keras.models.load_model(model_path)
        print(f"{model_type.upper()} model loaded from {model_path}")
    elif model_type in ['bert', 'deberta', 'albert', 'xlnet']:
        # Load models from HuggingFace transformers for BERT-like models
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        print(f"{model_type.upper()} model loaded from {model_path}")
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    return model


# Preprocessing function (specific for each model type)
def preprocess_text(text, tokenizer, max_length=50, model_type='cnn'):
    if model_type in ['bert', 'deberta', 'albert', 'xlnet']:
        # For transformer-based models, use the model-specific tokenizer
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=max_length)
        return inputs
    else:
        # For CNN, LSTM or other Keras models
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length)
        return padded_sequence


# Mapping of numeric labels to sentiment descriptions
LABEL_MAPPING = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive"
}

def decode_predictions(predictions, label_encoder):
    """
    Decode the model's predictions to a human-readable label.
    """
    # Print raw predictions for debugging
    print(f"Raw predictions: {predictions}")

    # Find the predicted class (highest probability)
    predicted_class = np.argmax(predictions, axis=-1)

    # Map the predicted class to the human-readable label
    predicted_label = LABEL_MAPPING.get(predicted_class[0], "Unknown")

    print(f"Predicted class index: {predicted_class[0]}")
    print(f"Decoded label: {predicted_label}")

    return predicted_label


# Main function to make predictions
def predict(model, tokenizer, label_encoder, text, model_type="cnn"):
    # Define max lengths for different models
    max_lengths = {
        "cnn": 32,
        "lstm": 50,
        "bert": 128,
        "deberta": 128,
        "albert": 128,
        "xlnet": 128,
    }

    max_length = max_lengths.get(model_type, 32)

    processed_text = preprocess_text(text, tokenizer, max_length, model_type)

    predictions = model.predict(processed_text)
    predicted_class = predictions.argmax(axis=-1)

    decoded_label = label_encoder.inverse_transform([predicted_class[0]])[0]
    return decoded_label


# Command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description="Run inference with a specified model")
    parser.add_argument('--text', type=str, required=True, help='Text to classify')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (e.g., CNN.keras, LSTM.keras)')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer file (e.g., tokenizer.pkl)')
    parser.add_argument('--embedding_matrix_path', type=str, required=False, help='Path to the embedding matrix file (e.g., embedding_matrix.npy)')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'lstm', 'bert', 'deberta', 'albert', 'xlnet'], required=True, help='Type of model to use')

    args = parser.parse_args()

    # Load the model and tokenizer
    model = load_model(args.model_path, args.model_type)
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # Load embedding matrix if it's a CNN, LSTM, or other Keras model
    if args.model_type in ['cnn', 'lstm'] and args.embedding_matrix_path:
        embedding_matrix = load_embedding_matrix(args.embedding_matrix_path)
    else:
        embedding_matrix = None

    # LabelEncoder (assuming the labels are integers: 0, 1, 2, 3, 4)
    label_encoder = LabelEncoder()
    label_encoder.fit([0, 1, 2, 3, 4])  # Fit the label encoder with your class labels

    # Make predictions on the provided text
    predicted_label = predict(model, tokenizer, label_encoder, args.text, model_type=args.model_type)
    print(f"Predicted Label: {predicted_label}")


if __name__ == '__main__':
    main()