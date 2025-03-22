import numpy as np
import tensorflow as tf
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import csv
import string

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

nltk.download("punkt", quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s.,!?]', '', text)
    tokens = word_tokenize(text)
    return " ".join(tokens)

def extract_features(text):
    if not isinstance(text, str):
        return [0, 0, 0, 0]
        
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    words = re.findall(r'\b\w+\b', text.lower())
    lexical_diversity = len(set(words)) / len(words) if words else 0
    
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    punctuation_freq = punctuation_count / len(text) if text else 0
    
    first_person = len(re.findall(r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b', text, re.IGNORECASE))
    first_person_freq = first_person / len(words) if words else 0
    
    return [avg_sentence_length, lexical_diversity, punctuation_freq, first_person_freq]

def load_model(model_type="lstm"):
    model_path = f"trained_models/tensorflow/{model_type}_model.h5"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded {model_type} model")
        return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None

def load_tokenizer_and_metadata():
    try:
        if os.path.exists("improved_data/tokenizer.pkl"):
            with open("improved_data/tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
            
            with open("improved_data/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            
            print("Loaded tokenizer and metadata from improved_data")
            return tokenizer, metadata
        
        elif os.path.exists("preprocessed_tf/tokenizer.pkl"):
            with open("preprocessed_tf/tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
            
            with open("preprocessed_tf/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            
            print("Loaded tokenizer and metadata from preprocessed_tf")
            return tokenizer, metadata
        
        else:
            print("Could not find tokenizer and metadata files")
            return None, None
    
    except Exception as e:
        print(f"Error loading tokenizer and metadata: {e}")
        return None, None

def load_datasets():
    inputs_path = "./datasets/dataset1_inputs.csv"
    outputs_path = "./datasets/dataset1_outputs.csv"
    
    try:
        print(f"Loading input texts from {inputs_path}...")
        inputs_data = []
        
        with open(inputs_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) > 0:
                    text = ','.join(row[1:]) if len(row) > 1 else row[0]
                    inputs_data.append(text)
        
        print(f"Loading output labels from {outputs_path}...")
        outputs_data = {}
        
        with open(outputs_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row and len(row) > 0:
                    parts = row[0].split('\t') if '\t' in row[0] else row
                    if len(parts) >= 2:
                        id_val = parts[0].strip()
                        label = parts[1].strip()
                        outputs_data[id_val] = 1 if label == "AI" else 0
        
        print(f"Loaded {len(inputs_data)} input texts and {len(outputs_data)} output labels")
        return inputs_data, outputs_data
    
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return [], {}

def predict_with_model(model, text, tokenizer, metadata, model_type="lstm"):
    if model is None or tokenizer is None or metadata is None:
        return None, None
    
    max_seq_length = metadata["max_seq_length"]
    
    if model_type == "dnn":
        features = np.array([extract_features(text)])
        prediction = model.predict(features, verbose=0)[0][0]
        binary_prediction = 1 if prediction >= 0.5 else 0
        return binary_prediction, prediction
    
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding='post', truncating='post')
    
    if model_type == "ensemble":
        num_inputs = len(model.inputs)
        
        inputs = []
        for i in range(num_inputs):
            input_shape = model.inputs[i].shape
            
            if len(input_shape) == 2 and input_shape[1] == 4:
                inputs.append(np.array([extract_features(text)]))
            else:
                inputs.append(padded_sequence)
        
        prediction = model.predict(inputs, verbose=0)[0][0]
    else:
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    binary_prediction = 1 if prediction >= 0.5 else 0
    
    return binary_prediction, prediction

def evaluate_models():
    tokenizer, metadata = load_tokenizer_and_metadata()
    if tokenizer is None or metadata is None:
        print("Cannot proceed without tokenizer and metadata")
        return
    
    inputs_data, outputs_data = load_datasets()
    if not inputs_data or not outputs_data:
        print("Cannot proceed without datasets")
        return
    
    model_types = ["lstm", "gru", "transformer", "dnn", "ensemble"]
    models = {}
    
    for model_type in model_types:
        model = load_model(model_type)
        if model is not None:
            models[model_type] = model
    
    if not models:
        print("No models could be loaded")
        return
    
    results = {model_type: {"correct": 0, "total": 0, "predictions": []} for model_type in models}
    
    print("\nEvaluating models on dataset...")
    for i, text in enumerate(tqdm(inputs_data)):
        id_match = re.match(r'D1-(\d+)', text)
        if id_match:
            id_val = f"D1-{id_match.group(1)}"
            text = re.sub(r'D1-\d+\s*', '', text)
        else:
            id_val = f"D1-{i}"
        
        if id_val not in outputs_data:
            continue
        
        true_label = outputs_data[id_val]
        
        cleaned_text = clean_text(text)
        
        for model_type, model in models.items():
            binary_prediction, raw_prediction = predict_with_model(
                model, cleaned_text, tokenizer, metadata, model_type
            )
            
            if binary_prediction is None:
                continue
                
            results[model_type]["total"] += 1
            if binary_prediction == true_label:
                results[model_type]["correct"] += 1
            
            results[model_type]["predictions"].append({
                "id": id_val,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "true_label": "AI" if true_label == 1 else "Human",
                "predicted_label": "AI" if binary_prediction == 1 else "Human",
                "confidence": raw_prediction if binary_prediction == 1 else 1 - raw_prediction,
                "correct": binary_prediction == true_label
            })
    
    print("\n=== Model Evaluation Results ===")
    for model_type in models:
        correct = results[model_type]["correct"]
        total = results[model_type]["total"]
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n{model_type.upper()} Model:")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        human_correct = sum(1 for p in results[model_type]["predictions"] 
                          if p["true_label"] == "Human" and p["correct"])
        human_total = sum(1 for p in results[model_type]["predictions"] 
                         if p["true_label"] == "Human")
        human_accuracy = human_correct / human_total if human_total > 0 else 0
        
        ai_correct = sum(1 for p in results[model_type]["predictions"] 
                        if p["true_label"] == "AI" and p["correct"])
        ai_total = sum(1 for p in results[model_type]["predictions"] 
                      if p["true_label"] == "AI")
        ai_accuracy = ai_correct / ai_total if ai_total > 0 else 0
        
        print(f"Human accuracy: {human_accuracy:.4f} ({human_correct}/{human_total})")
        print(f"AI accuracy: {ai_accuracy:.4f} ({ai_correct}/{ai_total})")
    
    os.makedirs("evaluation", exist_ok=True)
    for model_type in models:
        with open(f"evaluation/{model_type}_dataset_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Text", "True Label", "Predicted Label", "Confidence", "Correct"])
            
            for p in results[model_type]["predictions"]:
                writer.writerow([
                    p["id"],
                    p["text"],
                    p["true_label"],
                    p["predicted_label"],
                    f"{p['confidence']:.4f}",
                    "Yes" if p["correct"] else "No"
                ])
    
    print("\nDetailed results saved to evaluation/ directory")

if __name__ == "__main__":
    evaluate_models()