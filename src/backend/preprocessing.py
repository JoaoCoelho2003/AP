import numpy as np
import re
import pickle
import os
import string
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from collections import Counter

nltk.download("punkt", quiet=True)

def clean_text(text, keep_punctuation=False):
    """Clean text with option to keep punctuation for better sequence modeling"""
    text = text.lower()
    if not keep_punctuation:
        text = re.sub(f"[{string.punctuation}]", "", text)
    return text

def tokenize_text(text):
    """Tokenize text into words"""
    return word_tokenize(text)

def preprocess_dataset(dataset, max_samples=12000, keep_punctuation=False):
    """Process dataset and return texts and labels"""
    texts, labels = [], []
    tokenized_texts = []
    
    for i, example in enumerate(dataset):
        if "human_text" in example and "ai_text" in example:
            human_text = clean_text(example["human_text"], keep_punctuation)
            texts.append(human_text)
            tokenized_texts.append(tokenize_text(human_text))
            labels.append(0)
            
            ai_text = clean_text(example["ai_text"], keep_punctuation)
            texts.append(ai_text)
            tokenized_texts.append(tokenize_text(ai_text))
            labels.append(1)
            
        if len(texts) >= max_samples:
            break
    
    return texts, tokenized_texts, np.array(labels)

def build_vocabulary(tokenized_texts, max_vocab_size=10000):
    all_tokens = [token for text in tokenized_texts for token in text]
    counter = Counter(all_tokens)
    vocab = ["<PAD>", "<UNK>"] + [word for word, _ in counter.most_common(max_vocab_size - 2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    return vocab, word_to_idx

def train_word_embeddings(tokenized_texts, embedding_dim=100, window=5, min_count=1):
    model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, 
                     window=window, min_count=min_count, workers=4)
    return model

def train_fasttext_embeddings(tokenized_texts, embedding_dim=100, window=5, min_count=1):
    model = FastText(sentences=tokenized_texts, vector_size=embedding_dim, 
                     window=window, min_count=min_count, workers=4)
    return model

def texts_to_sequences(tokenized_texts, word_to_idx, max_length=100):
    sequences = []
    for text in tokenized_texts:
        seq = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in text[:max_length]]
        if len(seq) < max_length:
            seq = seq + [word_to_idx["<PAD>"]] * (max_length - len(seq))
        sequences.append(seq)
    return np.array(sequences)

def create_embedding_matrix(word_to_idx, embedding_model, embedding_dim=100):
    vocab_size = len(word_to_idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in word_to_idx.items():
        if word in embedding_model.wv:
            embedding_matrix[idx] = embedding_model.wv[word]
        else:
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
            
    return embedding_matrix

def create_averaged_embeddings(tokenized_texts, embedding_model, max_features=5000):
    embedding_dim = embedding_model.vector_size
    X = np.zeros((len(tokenized_texts), embedding_dim))
    
    for i, tokens in enumerate(tokenized_texts):
        if tokens:
            token_embeddings = [embedding_model.wv[token] for token in tokens if token in embedding_model.wv]
            if token_embeddings:
                X[i] = np.mean(token_embeddings, axis=0)
    
    return X

def main():
    print("Loading dataset...")
    dataset = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)
    
    print("Preprocessing texts...")
    raw_texts, tokenized_texts, labels = preprocess_dataset(dataset, max_samples=1000, keep_punctuation=True)
    
    print("Building vocabulary...")
    vocab, word_to_idx = build_vocabulary(tokenized_texts)
    
    print("Training word embeddings...")
    embedding_model = train_fasttext_embeddings(tokenized_texts)
    
    print("Creating sequence data for RNN...")
    max_length = 100
    X_sequences = texts_to_sequences(tokenized_texts, word_to_idx, max_length)
    embedding_matrix = create_embedding_matrix(word_to_idx, embedding_model)
    
    print("Creating averaged embeddings for DNN/Logistic...")
    X_averaged = create_averaged_embeddings(tokenized_texts, embedding_model)
    
    print("Splitting data...")
    X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        X_sequences, labels, test_size=0.2, random_state=2025)
    
    X_train_avg, X_val_avg, _, _ = train_test_split(
        X_averaged, labels, test_size=0.2, random_state=2025)
    
    print("Saving processed data...")
    output_dir = "preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_train_seq.npy"), X_train_seq)
    np.save(os.path.join(output_dir, "X_val_seq.npy"), X_val_seq)
    np.save(os.path.join(output_dir, "embedding_matrix.npy"), embedding_matrix)
    
    np.save(os.path.join(output_dir, "X_train_avg.npy"), X_train_avg)
    np.save(os.path.join(output_dir, "X_val_avg.npy"), X_val_avg)
    
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    
    with open(os.path.join(output_dir, "word_to_idx.pkl"), "wb") as f:
        pickle.dump(word_to_idx, f)
    
    embedding_model.save(os.path.join(output_dir, "embedding_model.model"))
    
    print("Preprocessing complete. Data saved!")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sequence data shape: {X_train_seq.shape}")
    print(f"Averaged data shape: {X_train_avg.shape}")

if __name__ == "__main__":
    main()
