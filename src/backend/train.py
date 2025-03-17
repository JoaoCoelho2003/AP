import numpy as np
import sys
import os
from models.logistic_regression import LogisticRegression
from models.dnn.neuralnet import NeuralNetwork
from models.dnn.layers import DenseLayer, DropoutLayer, BatchNormalizationLayer
from models.dnn.metrics import mse, accuracy
from models.dnn.activation import SigmoidActivation, ReLUActivation
from models.dnn.losses import FocalLoss
from models.rnn.rnn import RNN
from models.rnn.optimizers import AdamOptimizer

np.random.seed(2025)

class DatasetWrapper:
    def __init__(self, X, y):
        self.X = X
        self.y = y

model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"

if model_type == "dnn":
    X_train = np.load("preprocessed/X_train.npy")
    y_train = np.load("preprocessed/y_train.npy")
    X_val = np.load("preprocessed/X_val.npy")
    y_val = np.load("preprocessed/y_val.npy")
    
    print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    model = NeuralNetwork(
        epochs=100, 
        batch_size=128,
        learning_rate=0.0001,
        momentum=0.9,
        verbose=True,
        loss=FocalLoss,
        metric=accuracy,
        early_stopping=True,
        patience=5,
        lr_scheduler=True,
        lr_decay=0.2,
        min_delta=0.0001,
        mixup_alpha=0.2
    )
    
    n_features = X_train.shape[1]
    
    model.add(DenseLayer(32, (n_features,), l2_reg=0.05))
    model.add(BatchNormalizationLayer())
    model.add(ReLUActivation())
    model.add(DropoutLayer(0.5))
    
    model.add(DenseLayer(16, l2_reg=0.05))
    model.add(BatchNormalizationLayer())
    model.add(ReLUActivation())
    model.add(DropoutLayer(0.5))
    
    model.add(DenseLayer(1, l2_reg=0.05))
    model.add(SigmoidActivation())

    dataset = DatasetWrapper(X_train, y_train)
    val_dataset = DatasetWrapper(X_val, y_val)

    model.fit(dataset, val_dataset)

elif model_type == "rnn":
    X_train_seq = np.load("preprocessed/X_train_seq.npy")
    y_train = np.load("preprocessed/y_train.npy")
    X_val_seq = np.load("preprocessed/X_val_seq.npy")
    y_val = np.load("preprocessed/y_val.npy")
    embedding_matrix = np.load("preprocessed/embedding_matrix.npy")
    
    print(f"Data shapes - X_train_seq: {X_train_seq.shape}, y_train: {y_train.shape}")
    print(f"Data shapes - X_val_seq: {X_val_seq.shape}, y_val: {y_val.shape}")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    y_train_rnn = y_train.reshape(-1, 1)
    y_val_rnn = y_val.reshape(-1, 1)
    
    model = RNN(
        n_units=64,
        input_shape=(X_train_seq.shape[0], X_train_seq.shape[1]),
        embedding_matrix=embedding_matrix
    )
    
    model.set_dropout(rate=0.5, use_dropout=True)
    
    model.initialize(AdamOptimizer(lr=0.0005, beta1=0.9, beta2=0.999))
    
    model.train(X_train_seq, y_train_rnn, X_val_seq, y_val_rnn, batch_size=16, epochs=15)

else:
    X_train = np.load("preprocessed/X_train.npy")
    y_train = np.load("preprocessed/y_train.npy")
    X_val = np.load("preprocessed/X_val.npy")
    y_val = np.load("preprocessed/y_val.npy")
    
    model = LogisticRegression(lr=0.01, epochs=300)
    model.fit(X_train, y_train)

if not os.path.exists("trained_models"):
    os.makedirs("trained_models")

model.save(f"trained_models/{model_type}_weights.npz")
print("Training completed. Model saved!")