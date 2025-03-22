import numpy as np
import re
import pickle
import os
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(r'[!#$%&()*+/<=>?@[\]^_`{|}~]', '', text)
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words("english")) - {'i', 'me', 'my', 'myself', 'we', 'our'}
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def load_hc3_dataset(max_samples=10000):
    print("Loading HC3 dataset...")
    
    dataset = load_dataset("Hello-SimpleAI/HC3", name="all")
    
    texts, labels = [], []
    human_count, ai_count = 0, 0
    
    for example in dataset['train']:
        if "human_answers" in example and example["human_answers"]:
            for human_answer in example["human_answers"]:
                if human_answer and len(human_answer.strip()) > 0:
                    texts.append(human_answer)
                    labels.append(0)
                    human_count += 1
                    if human_count + ai_count >= max_samples:
                        break
            
        if "chatgpt_answers" in example and example["chatgpt_answers"]:
            for ai_answer in example["chatgpt_answers"]:
                if ai_answer and len(ai_answer.strip()) > 0:
                    texts.append(ai_answer)
                    labels.append(1)
                    ai_count += 1
                    if human_count + ai_count >= max_samples:
                        break
                        
        if human_count + ai_count >= max_samples:
            break
    
    print(f"Loaded {len(texts)} texts with {human_count} human and {ai_count} AI samples")
    return texts, np.array(labels)

def extract_features(texts):
    features = []
    
    for text in texts:
        if not isinstance(text, str):
            features.append([0, 0, 0, 0])
            continue
            
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        words = re.findall(r'\b\w+\b', text.lower())
        lexical_diversity = len(set(words)) / len(words) if words else 0
        
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        punctuation_freq = punctuation_count / len(text) if text else 0
        
        first_person = len(re.findall(r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b', text, re.IGNORECASE))
        first_person_freq = first_person / len(words) if words else 0
        
        features.append([avg_sentence_length, lexical_diversity, punctuation_freq, first_person_freq])
    
    return np.array(features)

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

def create_tf_tokenizer(texts, max_words=20000):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def texts_to_sequences(tokenizer, texts, max_seq_length=100):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')
    return padded_sequences

def load_glove_embeddings(tokenizer, embedding_dim=100):
    print("Loading GloVe embeddings...")
    
    glove_path = f'glove.6B.{embedding_dim}d.txt'
    if not os.path.exists(glove_path):
        os.system(f'wget http://nlp.stanford.edu/data/glove.6B.zip')
        os.system('unzip glove.6B.zip')
    
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    print(f"Found {len(embeddings_index)} word vectors in GloVe")
    
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def prepare_data():
    print("Loading dataset...")
    texts, labels = load_hc3_dataset()
    print(f"Initial data - Class distribution: {{0: {np.sum(labels == 0)}, 1: {np.sum(labels == 1)}}}")
    
    texts, labels = balance_dataset(texts, labels)
    print(f"Balanced data - Class distribution: {{0: {np.sum(labels == 0)}, 1: {np.sum(labels == 1)}}}")
    
    print("Extracting additional features...")
    additional_features = extract_features(texts)
    
    train_texts, test_texts, train_labels, test_labels, train_features, test_features = train_test_split(
        texts, labels, additional_features, test_size=0.2, random_state=2025, stratify=labels
    )
    
    test_texts, val_texts, test_labels, val_labels, test_features, val_features = train_test_split(
        test_texts, test_labels, test_features, test_size=0.5, random_state=2025, stratify=test_labels
    )
    
    print("Creating tokenizer...")
    tokenizer = create_tf_tokenizer(train_texts)
    
    print("Converting texts to sequences...")
    max_seq_length = 200
    train_sequences = texts_to_sequences(tokenizer, train_texts, max_seq_length)
    test_sequences = texts_to_sequences(tokenizer, test_texts, max_seq_length)
    val_sequences = texts_to_sequences(tokenizer, val_texts, max_seq_length)
    
    embedding_dim = 100
    embedding_matrix = load_glove_embeddings(tokenizer, embedding_dim)
    
    output_dir = "preprocessed_tf"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving processed data...")
    np.save(os.path.join(output_dir, "train_sequences.npy"), train_sequences)
    np.save(os.path.join(output_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(output_dir, "train_features.npy"), train_features)
    np.save(os.path.join(output_dir, "test_sequences.npy"), test_sequences)
    np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)
    np.save(os.path.join(output_dir, "test_features.npy"), test_features)
    np.save(os.path.join(output_dir, "val_sequences.npy"), val_sequences)
    np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(output_dir, "val_features.npy"), val_features)
    np.save(os.path.join(output_dir, "embedding_matrix.npy"), embedding_matrix)
    
    with open(os.path.join(output_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    
    metadata = {
        "max_seq_length": max_seq_length,
        "embedding_dim": embedding_dim,
        "vocab_size": len(tokenizer.word_index) + 1,
        "additional_features": train_features.shape[1]
    }
    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    print("Data preparation complete!")
    return metadata

if __name__ == "__main__":
    prepare_data()