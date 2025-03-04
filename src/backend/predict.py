import numpy as np
import sys
from models.logistic_regression import LogisticRegression
from models.bag_of_words import BagOfWords

model = LogisticRegression()
model.load("models/logistic_weights.npz")

bow = BagOfWords()
bow.load("models/vocab.npy")

def evaluate_text(text):
    """
    Evaluate whether the given text is AI-generated or human-written.
    
    Args:
        text (str): The text to evaluate
        
    Returns:
        tuple: (prediction, confidence) where prediction is "AI" or "Human"
    """
    X_new = bow.transform([text])
    probability = model.predict_proba(X_new)[0]
    prediction = "AI" if probability >= 0.5 else "Human"
    return prediction, probability

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            with open(filename, "r", encoding="utf-8") as file:
                text = file.read().strip()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        text = input("Enter text to evaluate: ").strip()

    prediction, confidence = evaluate_text(text)
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")