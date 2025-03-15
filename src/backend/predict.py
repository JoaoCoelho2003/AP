import numpy as np
import sys
import pickle
import os
from models.logistic_regression import LogisticRegression
from models.dnn.neuralnet import NeuralNetwork
from models.rnn.rnn import RNN
from models.rnn.optimizers import AdamOptimizer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(model_type):
    if model_type == "dnn":
        model = NeuralNetwork()
        model.load("trained_models/dnn_weights.npz")
    elif model_type == "rnn":
        model = RNN(n_units=256, input_shape=(100, 50))
        model.initialize(AdamOptimizer())
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
        class DatasetWrapper:
            def __init__(self, X):
                self.X = X
        probability = model.predict(DatasetWrapper(X_new))[0][0]
    elif model_type == "rnn":
        seq_length = 50
        n_features = X_new.shape[1]
        
        n_timesteps = n_features // seq_length
        if n_features % seq_length != 0:
            n_timesteps += 1
        
        if n_features % seq_length != 0:
            pad_size = seq_length - (n_features % seq_length)
            X_new_padded = np.pad(X_new, ((0, 0), (0, pad_size)), 'constant')
        else:
            X_new_padded = X_new
        
        X_new_rnn = X_new_padded.reshape((X_new.shape[0], n_timesteps, seq_length))
        
        predictions = model.forward_propagation(X_new_rnn)
        probability = predictions[0, -1, 0]
        
    else:
        probability = model.predict_proba(X_new)[0]

    prediction = "AI" if probability >= 0.5 else "Human"
    return prediction, probability

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    model, vectorizer = load_model(model_type)

    while True:
        text = input("Enter text to evaluate (or 'q' to quit): ").strip()
        if text.lower() == 'q':
            break
            
        if not text:
            print("Please enter some text.")
            continue
            
        prediction, confidence = evaluate_text(model, vectorizer, text, model_type)
        print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")
