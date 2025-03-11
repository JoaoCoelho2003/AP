import numpy as np
import sys
import pickle
import os
import re
import string
from nltk.tokenize import word_tokenize
import nltk
from models.logistic_regression import LogisticRegression
from models.dnn.neuralnet import NeuralNetwork
from models.rnn.rnn import RNN
from gensim.models import Word2Vec, FastText

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def clean_text(text, keep_punctuation=False):
    text = text.lower()
    if not keep_punctuation:
        text = re.sub(f"[{string.punctuation}]", "", text)
    return text

def tokenize_text(text):
    return word_tokenize(text)

def load_model(model_type):
    print(f"Loading {model_type} model...")
    
    with open("preprocessed/word_to_idx.pkl", "rb") as f:
        word_to_idx = pickle.load(f)
    
    if model_type == "rnn":
        embedding_matrix = np.load("preprocessed/embedding_matrix.npy")
        
        model = RNN(embedding_matrix=embedding_matrix, hidden_size=128, dropout_rate=0.2, l2_reg=0.001)
        model.load("trained_models/rnn_weights.npz")
        
        return model, word_to_idx, None
    
    elif model_type == "dnn":
        try:
            embedding_model = FastText.load("preprocessed/embedding_model.model")
        except:
            print("Error loading embedding model. Make sure you've run improved_preprocessing.py first.")
            sys.exit(1)
        
        model = NeuralNetwork()
        model.load("trained_models/dnn_weights.npz")
        
        return model, word_to_idx, embedding_model
    
    else:
        try:
            embedding_model = FastText.load("preprocessed/embedding_model.model")
        except:
            print("Error loading embedding model. Make sure you've run improved_preprocessing.py first.")
            sys.exit(1)
        
        model = LogisticRegression()
        model.load("trained_models/logistic_weights.npz")
        
        return model, word_to_idx, embedding_model

def preprocess_for_rnn(text, word_to_idx, max_length=100):
    cleaned_text = clean_text(text, keep_punctuation=True)
    tokens = tokenize_text(cleaned_text)
    
    sequence = []
    for token in tokens[:max_length]:
        idx = word_to_idx.get(token, word_to_idx.get("<UNK>", 1))
        sequence.append(idx)
    
    if len(sequence) < max_length:
        sequence = sequence + [word_to_idx.get("<PAD>", 0)] * (max_length - len(sequence))
    
    return np.array([sequence])

def preprocess_for_dnn_logistic(text, embedding_model):
    cleaned_text = clean_text(text, keep_punctuation=False)
    tokens = tokenize_text(cleaned_text)
    
    token_embeddings = []
    for token in tokens:
        if token in embedding_model.wv:
            token_embeddings.append(embedding_model.wv[token])
    
    if token_embeddings:
        averaged_embedding = np.mean(token_embeddings, axis=0)
    else:
        averaged_embedding = np.zeros(embedding_model.vector_size)
    
    return np.array([averaged_embedding])

def evaluate_text(model, word_to_idx, embedding_model, text, model_type):
    if model_type == "rnn":
        X_new = preprocess_for_rnn(text, word_to_idx)
        probability = model.predict(X_new)
        
        if isinstance(probability, np.ndarray):
            probability = probability.item()
    
    elif model_type == "dnn":
        X_new = preprocess_for_dnn_logistic(text, embedding_model)
        probability = model.forward_propagation(X_new, training=False)[0][0]
    
    else:
        X_new = preprocess_for_dnn_logistic(text, embedding_model)
        probability = model.predict_proba(X_new)[0]
    
    prediction = "AI" if probability >= 0.5 else "Human"
    return prediction, probability

def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    
    valid_models = ["rnn", "dnn", "logistic"]
    if model_type not in valid_models:
        print(f"Invalid model type. Please choose from: {', '.join(valid_models)}")
        sys.exit(1)
    
    model, word_to_idx, embedding_model = load_model(model_type)
    
    while True:
        text = input("\nEnter text to evaluate (or 'q' to quit): ").strip()
        
        if text.lower() == 'q':
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        prediction, confidence = evaluate_text(model, word_to_idx, embedding_model, text, model_type)
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
        
        if confidence > 0.8:
            strength = "very confident"
        elif confidence > 0.65:
            strength = "moderately confident"
        else:
            strength = "somewhat uncertain"
        
        print(f"The model is {strength} that this text was written by {'an AI' if prediction == 'AI' else 'a human'}.")

if __name__ == "__main__":
    main()