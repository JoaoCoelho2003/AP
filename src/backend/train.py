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
from models.rnn.optimizers import SGD

class DatasetWrapper:
    def __init__(self, X, y):
        self.X = X
        self.y = y

X_train = np.load("preprocessed/X_train.npy")
y_train = np.load("preprocessed/y_train.npy")
X_val = np.load("preprocessed/X_val.npy")
y_val = np.load("preprocessed/y_val.npy")

model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"

if model_type == "dnn":
    model = NeuralNetwork(epochs=25, batch_size=128, learning_rate=0.001, verbose=True,
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
    n_timesteps = X_train.shape[1]
    X_train_rnn = X_train.reshape((X_train.shape[0], n_timesteps, 1))
    
    model = RNN(n_units=10, input_shape=(n_timesteps, 1))
    model.initialize(SGD())
    model.forward_propagation(X_train_rnn)
    model.backward_propagation(np.random.rand(*X_train_rnn.shape))

else:
    model = LogisticRegression(lr=0.01, epochs=300)
    model.fit(X_train, y_train)

if not os.path.exists("trained_models"):
    os.makedirs("trained_models")

model.save(f"trained_models/{model_type}_weights.npz")
print("Training completed. Model saved!")
