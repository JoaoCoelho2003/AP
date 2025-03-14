import numpy as np
import re
import pickle
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download("punkt")
nltk.download("stopwords")

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_dataset(dataset, batch_size=25000):
    texts, labels = [], []
    for i, example in enumerate(dataset):
        if "human_text" in example and "ai_text" in example:
            texts.append(clean_text(example["human_text"]))
            labels.append(0)
            texts.append(clean_text(example["ai_text"]))
            labels.append(1)
        if len(texts) >= batch_size:
            break
    return texts, np.array(labels)

def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()
    return X, vectorizer

dataset = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)
train_texts, train_labels = preprocess_dataset(dataset)

X_train, vectorizer = vectorize_texts(train_texts)

X_train, X_val, y_train, y_val = train_test_split(X_train, train_labels, test_size=0.2, random_state=2025)

output_dir = "preprocessed"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)

with open("preprocessed/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Preprocessing complete. Data saved!")