import numpy as np
import sys
import pickle
from models.logistic_regression import LogisticRegression
from models.dnn.neuralnet import NeuralNetwork
from models.rnn.rnn import RNN
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(model_type):
    if model_type == "dnn":
        model = NeuralNetwork()
        model.load("trained_models/dnn_weights.npz")
    elif model_type == "rnn":
        model = RNN(n_units=10, input_shape=(None, 1))
        model.load("trained_models/rnn_weights.npz")
    else:
        model = LogisticRegression()
        model.load("trained_models/logistic_weights.npz")

    with open("preprocessed/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def evaluate_text(model, vectorizer, text, model_type):
    X_new = vectorizer.transform([text]).toarray()
    
    if model_type == "dnn":
        probability = model.forward_propagation(X_new, training=False)[0][0]
    elif model_type == "rnn":
        X_new_rnn = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
        probability = model.predict(X_new_rnn)[0][0]
    else:
        probability = model.predict_proba(X_new)[0]

    prediction = "AI" if probability >= 0.5 else "Human"
    return prediction, probability

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    model, vectorizer = load_model(model_type)

    text = input("Enter text to evaluate: ").strip()
    prediction, confidence = evaluate_text(model, vectorizer, text, model_type)
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")
