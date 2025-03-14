import numpy as np
import sys
import os
from models.logistic_regression import LogisticRegression
from models.dnn.neuralnet import NeuralNetwork
from models.dnn.layers import DenseLayer, DropoutLayer
from models.dnn.metrics import mse, accuracy
from models.dnn.activation import SigmoidActivation, ReLUActivation
from models.dnn.losses import BinaryCrossEntropy
from models.rnn.rnn import RNN
from models.rnn.optimizers import SGD, AdamOptimizer

class DatasetWrapper:
    def __init__(self, X, y):
        self.X = X
        self.y = y

X_train = np.load("preprocessed/X_train.npy")
y_train = np.load("preprocessed/y_train.npy")
X_val = np.load("preprocessed/X_val.npy")
y_val = np.load("preprocessed/y_val.npy")

print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")

model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"

if model_type == "dnn":
    model = NeuralNetwork(epochs=25, batch_size=64, learning_rate=0.001, verbose=True,
                          loss=BinaryCrossEntropy, metric=accuracy)
    
    n_features = X_train.shape[1]
    model.add(DenseLayer(64, (n_features,)))
    model.add(ReLUActivation())
    model.add(DropoutLayer(0.2))
    
    model.add(DenseLayer(32))
    model.add(ReLUActivation())
    model.add(DropoutLayer(0.2))
    
    model.add(DenseLayer(1)) 
    model.add(SigmoidActivation())

    dataset = DatasetWrapper(X_train, y_train)
    val_dataset = DatasetWrapper(X_val, y_val)

    model.fit(dataset, val_dataset)

elif model_type == "rnn":
    seq_length = 50
    n_features = X_train.shape[1]
    
    n_timesteps = n_features // seq_length
    if n_features % seq_length != 0:
        n_timesteps += 1
    
    if n_features % seq_length != 0:
        pad_size = seq_length - (n_features % seq_length)
        X_train_padded = np.pad(X_train, ((0, 0), (0, pad_size)), 'constant')
        X_val_padded = np.pad(X_val, ((0, 0), (0, pad_size)), 'constant')
    else:
        X_train_padded = X_train
        X_val_padded = X_val
    
    X_train_rnn = X_train_padded.reshape((X_train.shape[0], n_timesteps, seq_length))
    X_val_rnn = X_val_padded.reshape((X_val.shape[0], n_timesteps, seq_length))
    
    y_train_rnn = y_train.reshape(-1, 1)
    y_val_rnn = y_val.reshape(-1, 1)
    
    print(f"RNN input shapes - X_train_rnn: {X_train_rnn.shape}, y_train_rnn: {y_train_rnn.shape}")
    print(f"RNN input shapes - X_val_rnn: {X_val_rnn.shape}, y_val_rnn: {y_val_rnn.shape}")
    
    model = RNN(n_units=256, input_shape=(n_timesteps, seq_length))
    model.initialize(AdamOptimizer(lr=0.0005, beta1=0.9, beta2=0.999))
    
    model.train(X_train_rnn, y_train_rnn, X_val_rnn, y_val_rnn, batch_size=16, epochs=15)

else:
    model = LogisticRegression(lr=0.01, epochs=300)
    model.fit(X_train, y_train)

if not os.path.exists("trained_models"):
    os.makedirs("trained_models")

model.save(f"trained_models/{model_type}_weights.npz")
print("Training completed. Model saved!")
