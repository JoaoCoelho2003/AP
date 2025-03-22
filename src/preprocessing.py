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
from gensim.models import Word2Vec

nltk.download("punkt")
nltk.download("stopwords")

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def preprocess_dataset(dataset, batch_size=5000, format_type="default"):
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

def vectorize_texts_for_tfidf(tokenized_texts):
    joined_texts = [" ".join(tokens) for tokens in tokenized_texts]
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(joined_texts).toarray()
    return X, vectorizer

def train_word2vec(tokenized_texts, vector_size=300, window=5, min_count=1, workers=4, epochs=10):
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )
    print(f"Vocabulary size: {len(model.wv.key_to_index)}")
    return model

def create_embedding_matrix(tokenized_texts, word_vectors, max_seq_length=100):
    all_words = set()
    for tokens in tokenized_texts:
        all_words.update(tokens)
    
    vocabulary = [word for word in all_words if word in word_vectors.key_to_index]
    print(f"Vocabulary size after filtering: {len(vocabulary)}")
    
    word_to_idx = {word: i+1 for i, word in enumerate(vocabulary)}
    
    embedding_dim = word_vectors.vector_size
    embedding_matrix = np.zeros((len(vocabulary) + 1, embedding_dim))
    
    for word, idx in word_to_idx.items():
        embedding_matrix[idx] = word_vectors[word]
    
    sequences = np.zeros((len(tokenized_texts), max_seq_length), dtype=int)
    
    for i, tokens in enumerate(tokenized_texts):
        for j, word in enumerate(tokens[:max_seq_length]):
            if word in word_to_idx:
                sequences[i, j] = word_to_idx[word]
    
    return sequences, word_to_idx, embedding_matrix

dataset = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)
train_tokenized_texts, train_labels = preprocess_dataset(dataset, format_type="train")

word2vec_model = train_word2vec(
    train_tokenized_texts,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    epochs=15
)

os.makedirs("models", exist_ok=True)
word2vec_model.save("models/word2vec_custom.model")
print("Custom Word2Vec model saved")

X_train, vectorizer = vectorize_texts_for_tfidf(train_tokenized_texts)
print(f"TF-IDF features: {len(vectorizer.get_feature_names_out())}")

max_seq_length = 100
X_train_seq, word_to_idx, embedding_matrix = create_embedding_matrix(
    train_tokenized_texts, word2vec_model.wv, max_seq_length)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

validation_dataset = load_dataset("andythetechnerd03/AI-human-text", split="train", streaming=True)
val_tokenized_texts, val_labels = preprocess_dataset(validation_dataset, batch_size=2000, format_type="validation")

print(f"Validation dataset before balancing - Class distribution: {{0: {np.sum(val_labels == 0)}, 1: {np.sum(val_labels == 1)}}}")
val_tokenized_texts, val_labels = balance_dataset(val_tokenized_texts, val_labels)
print(f"Validation dataset after balancing - Class distribution: {{0: {np.sum(val_labels == 0)}, 1: {np.sum(val_labels == 1)}}}")

val_joined_texts = [" ".join(tokens) for tokens in val_tokenized_texts]
X_val = vectorizer.transform(val_joined_texts).toarray()

X_val_seq = np.zeros((len(val_tokenized_texts), max_seq_length), dtype=int)
for i, tokens in enumerate(val_tokenized_texts):
    for j, word in enumerate(tokens[:max_seq_length]):
        if word in word_to_idx:
            X_val_seq[i, j] = word_to_idx[word]

output_dir = "preprocessed"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), train_labels)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), val_labels)

with open("preprocessed/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

np.save(os.path.join(output_dir, "X_train_seq.npy"), X_train_seq)
np.save(os.path.join(output_dir, "X_val_seq.npy"), X_val_seq)
np.save(os.path.join(output_dir, "embedding_matrix.npy"), embedding_matrix)

with open("preprocessed/word_to_idx.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)

print("Preprocessing complete. Data saved!")
