import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models.tensorflowModels.dnn_model import train_dnn_model
from models.tensorflowModels.rnn_model import train_rnn_model
from models.tensorflowModels.transformer_model import train_transformer_model
from models.tensorflowModels.bert_model import train_bert_model
from models.tensorflowModels.ensemble_model import train_ensemble_model

def load_data(data_dir="preprocessed_tf"):
    train_sequences = np.load(os.path.join(data_dir, "train_sequences.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    test_sequences = np.load(os.path.join(data_dir, "test_sequences.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    val_sequences = np.load(os.path.join(data_dir, "val_sequences.npy"))
    val_labels = np.load(os.path.join(data_dir, "val_labels.npy"))

    embedding_matrix = np.load(os.path.join(data_dir, "embedding_matrix.npy"))

    with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    with open(os.path.join(data_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    try:
        with open(os.path.join(data_dir, "train_texts.pkl"), "rb") as f:
            train_texts = pickle.load(f)
        with open(os.path.join(data_dir, "val_texts.pkl"), "rb") as f:
            val_texts = pickle.load(f)
        with open(os.path.join(data_dir, "test_texts.pkl"), "rb") as f:
            test_texts = pickle.load(f)
    except:
        train_texts = None
        val_texts = None
        test_texts = None

    try:
        train_features = np.load(os.path.join(data_dir, "train_features.npy"))
        test_features = np.load(os.path.join(data_dir, "test_features.npy"))
        val_features = np.load(os.path.join(data_dir, "val_features.npy"))
    except:
        train_features = None
        test_features = None
        val_features = None

    return {
        "train_sequences": train_sequences,
        "train_labels": train_labels,
        "test_sequences": test_sequences,
        "test_labels": test_labels,
        "val_sequences": val_sequences,
        "val_labels": val_labels,
        "embedding_matrix": embedding_matrix,
        "metadata": metadata,
        "tokenizer": tokenizer,
        "train_texts": train_texts,
        "val_texts": val_texts,
        "test_texts": test_texts,
        "train_features": train_features,
        "test_features": test_features,
        "val_features": val_features
    }

def evaluate_model(model, X_test, y_test, model_name):
    os.makedirs("evaluation", exist_ok=True)

    y_pred_prob = model.predict(X_test, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"evaluation/{model_name}_confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": (y_pred == y_test).mean(),
        "auc": roc_auc,
    }

def train_models():
    data = load_data()

    train_sequences = data["train_sequences"]
    train_labels = data["train_labels"]
    test_sequences = data["test_sequences"]
    test_labels = data["test_labels"]
    val_sequences = data["val_sequences"]
    val_labels = data["val_labels"]
    embedding_matrix = data["embedding_matrix"]
    metadata = data["metadata"]
    tokenizer = data["tokenizer"]
    train_texts = data["train_texts"]
    val_texts = data["val_texts"]
    test_texts = data["test_texts"]
    train_features = data["train_features"]
    test_features = data["test_features"]
    val_features = data["val_features"]

    vocab_size = metadata["vocab_size"]
    embedding_dim = metadata["embedding_dim"]
    max_seq_length = metadata["max_seq_length"]

    os.makedirs("models", exist_ok=True)

    print("\n=== Training LSTM Model ===")
    lstm_model, lstm_history = train_rnn_model(
        train_sequences, train_labels,
        val_sequences, val_labels,
        vocab_size, embedding_dim, embedding_matrix, max_seq_length,
        model_type='lstm',
        model_path='trained_models/tensorflow/lstm_model.h5'
    )
    lstm_metrics = evaluate_model(lstm_model, test_sequences, test_labels, "LSTM")
    print(f"LSTM Model - Test Accuracy: {lstm_metrics['accuracy']:.4f}, AUC: {lstm_metrics['auc']:.4f}")

    print("\n=== Training GRU Model ===")
    gru_model, gru_history = train_rnn_model(
        train_sequences, train_labels,
        val_sequences, val_labels,
        vocab_size, embedding_dim, embedding_matrix, max_seq_length,
        model_type='gru',
        model_path='trained_models/tensorflow/gru_model.h5'
    )
    gru_metrics = evaluate_model(gru_model, test_sequences, test_labels, "GRU")
    print(f"GRU Model - Test Accuracy: {gru_metrics['accuracy']:.4f}, AUC: {gru_metrics['auc']:.4f}")

    print("\n=== Training Transformer Model ===")
    transformer_model, transformer_history = train_transformer_model(
        train_sequences, train_labels,
        val_sequences, val_labels,
        vocab_size, embedding_dim, embedding_matrix, max_seq_length,
        model_path='trained_models/tensorflow/transformer_model.h5'
    )
    transformer_metrics = evaluate_model(transformer_model, test_sequences, test_labels, "Transformer")
    print(f"Transformer Model - Test Accuracy: {transformer_metrics['accuracy']:.4f}, AUC: {transformer_metrics['auc']:.4f}")

    if train_features is not None:
        print("\n=== Training DNN Model ===")
        dnn_model, dnn_history = train_dnn_model(
            train_features, train_labels,
            val_features, val_labels,
            model_path='trained_models/tensorflow/dnn_model.h5'
        )
        dnn_metrics = evaluate_model(dnn_model, test_features, test_labels, "DNN")
        print(f"DNN Model - Test Accuracy: {dnn_metrics['accuracy']:.4f}, AUC: {dnn_metrics['auc']:.4f}")
    else:
        print("\n=== Skipping DNN Model (no features available) ===")
        dnn_model = None
        dnn_metrics = {"accuracy": 0, "auc": 0}

    print("\n=== Training Ensemble Model ===")
    models_to_ensemble = [m for m in [lstm_model, gru_model, transformer_model, dnn_model] if m is not None]
    
    if len(models_to_ensemble) > 1:
        X_trains = []
        X_vals = []
        X_tests = []
        
        for model in models_to_ensemble:
            if model == dnn_model and dnn_model is not None:
                X_trains.append(train_features)
                X_vals.append(val_features)
                X_tests.append(test_features)
            else:
                X_trains.append(train_sequences)
                X_vals.append(val_sequences)
                X_tests.append(test_sequences)

        ensemble_model, ensemble_history = train_ensemble_model(
            X_trains, train_labels,
            X_vals, val_labels,
            models_to_ensemble,
            model_path='trained_models/tensorflow/ensemble_model.h5'
        )
        ensemble_metrics = evaluate_model(ensemble_model, X_tests, test_labels, "Ensemble")
        print(f"Ensemble Model - Test Accuracy: {ensemble_metrics['accuracy']:.4f}, AUC: {ensemble_metrics['auc']:.4f}")
    else:
        print("Not enough models to create an ensemble.")
        ensemble_metrics = {"accuracy": 0, "auc": 0}

    print("\n=== Model Performance Summary ===")
    print(f"LSTM Model - Accuracy: {lstm_metrics['accuracy']:.4f}, AUC: {lstm_metrics['auc']:.4f}")
    print(f"GRU Model - Accuracy: {gru_metrics['accuracy']:.4f}, AUC: {gru_metrics['auc']:.4f}")
    print(f"Transformer Model - Accuracy: {transformer_metrics['accuracy']:.4f}, AUC: {transformer_metrics['auc']:.4f}")
    if dnn_model is not None:
        print(f"DNN Model - Accuracy: {dnn_metrics['accuracy']:.4f}, AUC: {dnn_metrics['auc']:.4f}")
    if len(models_to_ensemble) > 1:
        print(f"Ensemble Model - Accuracy: {ensemble_metrics['accuracy']:.4f}, AUC: {ensemble_metrics['auc']:.4f}")

if __name__ == "__main__":
    train_models()