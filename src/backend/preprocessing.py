import numpy as np
import re
import pickle
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.utils import shuffle

nltk.download("punkt")
nltk.download("stopwords")

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_dataset(dataset, batch_size=25000, format_type="default"):
    texts, labels = [], []
    for i, example in enumerate(dataset):
        if format_type == "train":
            if "human_text" in example and "ai_text" in example:
                texts.append(clean_text(example["human_text"]))
                labels.append(0)
                texts.append(clean_text(example["ai_text"]))
                labels.append(1)
        elif format_type == "validation":
            if "text" in example and "generated" in example:
                texts.append(clean_text(example["text"]))
                labels.append(example["generated"])
        if len(texts) >= batch_size:
            break
    return texts, np.array(labels)

def balance_dataset(texts, labels):
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]
    
    min_class_size = min(len(class_0_indices), len(class_1_indices))
    
    if len(class_0_indices) > min_class_size:
        class_0_indices = np.random.choice(class_0_indices, min_class_size, replace=False)
    if len(class_1_indices) > min_class_size:
        class_1_indices = np.random.choice(class_1_indices, min_class_size, replace=False)
    
    balanced_indices = np.concatenate([class_0_indices, class_1_indices])
    balanced_indices = shuffle(balanced_indices)
    
    return [texts[i] for i in balanced_indices], labels[balanced_indices]

def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()
    return X, vectorizer

dataset = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)
train_texts, train_labels = preprocess_dataset(dataset, format_type="train")
X_train, vectorizer = vectorize_texts(train_texts)
print(len(vectorizer.get_feature_names_out()))


validation_dataset = load_dataset("andythetechnerd03/AI-human-text", split="train", streaming=True)
val_texts, val_labels = preprocess_dataset(validation_dataset, batch_size=10000, format_type="validation")

print(f"Validation dataset before balancing - Class distribution: {{0: {np.sum(val_labels == 0)}, 1: {np.sum(val_labels == 1)}}}")
val_texts, val_labels = balance_dataset(val_texts, val_labels)
print(f"Validation dataset after balancing - Class distribution: {{0: {np.sum(val_labels == 0)}, 1: {np.sum(val_labels == 1)}}}")

X_val = vectorizer.transform(val_texts).toarray()

output_dir = "preprocessed"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), train_labels)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), val_labels)

with open("preprocessed/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Preprocessing complete. Data saved!")