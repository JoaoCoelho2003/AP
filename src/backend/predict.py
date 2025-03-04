import numpy as np
import sys
from models.logistic_regression import LogisticRegression
from models.bag_of_words import BagOfWords
from models.dnn.neuralnet import NeuralNetwork

def load_model(model_type):
    bow = BagOfWords()
    if model_type == "neuralnet":
        model = NeuralNetwork()
        model.load("trained_models/neuralnet_weights.npz")
        bow.load("trained_models/vocab_dnn.npy")
    else:
        model = LogisticRegression()
        model.load("trained_models/logistic_weights.npz")
        bow.load("trained_models/vocab_log.npy")
        
    return model, bow

def evaluate_text(model, bow, text, model_type):
    """
    Evaluate whether the given text is AI-generated or human-written.
    
    Args:
        model: The model to use for prediction (LogisticRegression or NeuralNetwork)
        bow: The BagOfWords instance
        text (str): The text to evaluate
        model_type (str): The type of model being used
        
    Returns:
        tuple: (prediction, confidence) where prediction is "AI" or "Human"
    """
    X_new = bow.transform([text])
    if(model_type == "neuralnet"):
        probability = model.forward_propagation(X_new, training=False)[0][0]
    else:
        probability = model.predict_proba(X_new)[0]
        
    prediction = "AI" if probability >= 0.5 else "Human"
    return prediction, probability

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    model, bow = load_model(model_type)
    
    text = input("Enter text to evaluate: ").strip()
    prediction, confidence = evaluate_text(model, bow, text, model_type)
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")