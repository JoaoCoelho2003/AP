import numpy as np
import sys
import pickle
from models.logistic_regression import LogisticRegression
from models.dnn.neuralnet import NeuralNetwork
from models.rnn.rnn import RNN
from models.rnn.optimizers import AdamOptimizer
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def load_model(model_type):
    if model_type == "dnn":
        model = NeuralNetwork()
        model.load("trained_models/dnn_weights.npz")
        
        with open("preprocessed/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer, None, None
    
    elif model_type == "rnn":
        embedding_matrix = np.load("preprocessed/embedding_matrix.npy")
        
        model = RNN(n_units=64, embedding_matrix=embedding_matrix)
        model.initialize(AdamOptimizer())
        model.load("trained_models/rnn_weights.npz")
        
        with open("preprocessed/word_to_idx.pkl", "rb") as f:
            word_to_idx = pickle.load(f)
        
        return model, None, word_to_idx, embedding_matrix
    
    else:
        model = LogisticRegression()
        model.load("trained_models/logistic_weights.npz")
        
        with open("preprocessed/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer, None, None

def evaluate_text(model, vectorizer, word_to_idx, embedding_matrix, text, model_type):
    tokens = clean_text(text)
    
    if model_type == "rnn":
        max_seq_length = 100
        sequence = np.zeros((1, max_seq_length), dtype=int)
        
        for j, word in enumerate(tokens[:max_seq_length]):
            if word in word_to_idx:
                sequence[0, j] = word_to_idx[word]
        
        predictions = model.forward_propagation(sequence)
        probability = predictions[0, -1, 0]
    
    else:
        joined_text = " ".join(tokens)
        X_new = vectorizer.transform([joined_text]).toarray()
        
        if model_type == "dnn":
            class DatasetWrapper:
                def __init__(self, X):
                    self.X = X
            probability = model.predict(DatasetWrapper(X_new))[0][0]
        else:
            probability = model.predict_proba(X_new)[0]

    prediction = "AI" if probability >= 0.5 else "Human"
    return prediction, probability

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    model, vectorizer, word_to_idx, embedding_matrix = load_model(model_type)

    while True:
        text = input("Enter text to evaluate (or 'q' to quit): ").strip()
        if text.lower() == 'q':
            break
            
        if not text:
            print("Please enter some text.")
            continue
            
        prediction, confidence = evaluate_text(model, vectorizer, word_to_idx, embedding_matrix, text, model_type)
        print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")
