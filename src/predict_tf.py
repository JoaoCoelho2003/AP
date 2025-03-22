import numpy as np
import tensorflow as tf
import pickle
import os
import re
import string
from nltk.tokenize import word_tokenize
import nltk
import argparse
import warnings
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

nltk.download("punkt", quiet=True)

def clean_text(text):
    text = text.lower()
    
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    tokens = word_tokenize(text)
    
    return " ".join(tokens)

def load_model(model_type="lstm"):
    model_path = f"trained_models/tensorflow/{model_type}_model.h5"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    with open("preprocessed_tf/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    with open("preprocessed_tf/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return model, tokenizer, metadata

def predict_text(text, model_type="lstm", explain=False):
    cleaned_text = clean_text(text)
    
    try:
        model, tokenizer, metadata = load_model(model_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running train_tf.py")
        return None, None, None
    
    max_seq_length = metadata["max_seq_length"]
    
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=max_seq_length, padding='post', truncating='post'
    )
    
    if model_type == "ensemble":
        num_models = len(model.inputs)
        inputs = [padded_sequence] * num_models
        prediction = model.predict(inputs, verbose=0)[0][0]
    else:
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    predicted_class = "AI" if prediction >= 0.5 else "Human"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
        
    return predicted_class, confidence

def main():
	model_type = sys.argv[1] if len(sys.argv) > 1 else "lstm"
	
	try:
		model, tokenizer, metadata = load_model(model_type)
	except FileNotFoundError as e:
		print(f"Error: {e}")
		print("Please train the model first by running train_tf.py")
		return
	
	while True:
		text = input("Enter text to classify (or 'q' to quit): ").strip()
		if text.lower() == 'q':
			break
		
		if not text:
			print("Please enter some text.")
			continue
		
		predicted_class, confidence = predict_text(text, model_type)
		
		if predicted_class is None:
			continue
		
		print(f"\nPrediction: {predicted_class}")
		print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
	main()

if __name__ == "__main__":
    main()