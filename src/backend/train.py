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
from models.rnn.optimizers import AdamOptimizer

class DatasetWrapper:
    def __init__(self, X, y):
        self.X = X
        self.y = y

def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    
    print(f"Training model: {model_type}")
    
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    
    if model_type == "rnn":
        X_train = np.load("preprocessed/X_train_seq.npy")
        y_train = np.load("preprocessed/y_train.npy")
        X_val = np.load("preprocessed/X_val_seq.npy")
        y_val = np.load("preprocessed/y_val.npy")
        embedding_matrix = np.load("preprocessed/embedding_matrix.npy")
        
        print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        
        model = RNN(
            embedding_matrix=embedding_matrix,
            hidden_size=128,
            dropout_rate=0.2,
            l2_reg=0.001
        )
        
        model.initialize(AdamOptimizer(lr=0.0001, beta1=0.9, beta2=0.999))
        model.train(X_train, y_train, X_val, y_val, batch_size=64, epochs=50)
    
    elif model_type == "dnn":
        X_train = np.load("preprocessed/X_train_avg.npy")
        y_train = np.load("preprocessed/y_train.npy")
        X_val = np.load("preprocessed/X_val_avg.npy")
        y_val = np.load("preprocessed/y_val.npy")
        
        print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        model = NeuralNetwork(epochs=25, batch_size=64, learning_rate=0.001, verbose=True,
                              loss=BinaryCrossEntropy, metric=accuracy)
        
        n_features = X_train.shape[1]
        model.add(DenseLayer(128, (n_features,)))
        model.add(ReLUActivation())
        model.add(DropoutLayer(0.3))
        
        model.add(DenseLayer(64))
        model.add(ReLUActivation())
        model.add(DropoutLayer(0.3))
        
        model.add(DenseLayer(32))
        model.add(ReLUActivation())
        model.add(DropoutLayer(0.2))
        
        model.add(DenseLayer(1)) 
        model.add(SigmoidActivation())
        
        dataset = DatasetWrapper(X_train, y_train)
        val_dataset = DatasetWrapper(X_val, y_val)
        
        model.fit(dataset, val_dataset)
    
    elif model_type == "logistic":
        X_train = np.load("preprocessed/X_train_avg.npy")
        y_train = np.load("preprocessed/y_train.npy")
        X_val = np.load("preprocessed/X_val_avg.npy")
        y_val = np.load("preprocessed/y_val.npy")
        
        print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        model = LogisticRegression(lr=0.01, epochs=300)
        model.fit(X_train, y_train)
        
        val_preds = model.predict(X_val)
        val_accuracy = np.mean((val_preds == y_val).astype(float))
        print(f"Validation accuracy: {val_accuracy:.4f}")
    
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    model.save(f"trained_models/{model_type}_weights.npz")
    print("Training completed. Model saved!")

if __name__ == "__main__":
    main()

