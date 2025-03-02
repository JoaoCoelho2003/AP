import numpy as np
import sys
from models.logistic_regression import LogisticRegression
from models.bag_of_words import BagOfWords

model = LogisticRegression()
model.load("models/logistic_weights.npz")

bow = BagOfWords()
bow.load("models/vocab.npy")

def evaluate_text(text):
    X_new = bow.transform([text])
    prediction = model.predict(X_new)
    return "AI" if prediction[0] == 1 else "Human"

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

print(f"Prediction: {evaluate_text(text)}")
