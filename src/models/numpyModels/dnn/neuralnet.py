import os
import numpy as np
import copy
from scipy.optimize import minimize_scalar

from models.numpyModels.dnn.layers import DenseLayer, DropoutLayer
from models.numpyModels.dnn.losses import LossFunction, MeanSquaredError
from models.numpyModels.dnn.optimizer import Optimizer
from models.numpyModels.dnn.metrics import mse

class NeuralNetwork:
    def __init__(self, epochs=100, batch_size=128, optimizer=None,
                 learning_rate=0.01, momentum=0.90, verbose=False, 
                 loss=MeanSquaredError, metric=mse,
                 early_stopping=True, patience=5, lr_scheduler=True, lr_decay=0.5,
                 min_delta=0.001, mixup_alpha=0.2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.early_stopping = early_stopping
        self.patience = patience
        self.lr_scheduler = lr_scheduler
        self.lr_decay = lr_decay
        self.min_delta = min_delta
        self.mixup_alpha = mixup_alpha

        self.layers = []
        self.history = {}
        self.best_val_loss = np.inf
        self.no_improvement_count = 0
        self.best_weights = None
        self.temperature = 1.0

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def mixup_data(self, X, y):
        batch_size = X.shape[0]
        indices = np.random.permutation(batch_size)
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, batch_size)
        lam = np.maximum(lam, 1 - lam)
        
        lam_x = lam.reshape(-1, 1)
        lam_y = lam.reshape(-1, 1)
        
        mixed_X = lam_x * X + (1 - lam_x) * X[indices]
        mixed_y = lam_y * y + (1 - lam_y) * y[indices]
        
        return mixed_X, mixed_y

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error
    
    def save_model_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'get_weights'):
                weights.append(layer.get_weights())
            else:
                weights.append(None)
        return weights
    
    def restore_model_weights(self, weights):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_weights') and weights[i] is not None:
                layer.set_weights(weights[i])

    def calibrate_temperature(self, X_val, y_val):
        raw_preds = self.forward_propagation(X_val, training=False)
        
        def nll_loss(T):
            logits = np.log(raw_preds / (1 - raw_preds + 1e-15))
            scaled_preds = 1 / (1 + np.exp(-logits / T))
            scaled_preds = np.clip(scaled_preds, 1e-15, 1 - 1e-15)
            nll = -np.mean(y_val * np.log(scaled_preds) + (1 - y_val) * np.log(1 - scaled_preds))
            return nll
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        if self.verbose:
            print(f"Calibrated temperature: {self.temperature:.4f}")

    def fit(self, dataset, val_dataset=None):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        self.best_val_loss = np.inf
        self.no_improvement_count = 0
        
        for epoch in range(1, self.epochs + 1):
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                if self.mixup_alpha > 0:
                    X_batch, y_batch = self.mixup_data(X_batch, y_batch)
                
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            self.history[epoch] = {'loss': loss, 'metric': metric}

            val_loss, val_metric = None, None
            if val_dataset:
                val_X, val_y = val_dataset.X, val_dataset.y
                if np.ndim(val_y) == 1:
                    val_y = np.expand_dims(val_y, axis=1)

                val_output = self.forward_propagation(val_X, training=False)
                val_loss = self.loss.loss(val_y, val_output)

                if self.metric is not None:
                    val_metric = self.metric(val_y, val_output)
                    val_metric_s = f"{self.metric.__name__}: {val_metric:.4f}"
                else:
                    val_metric_s = "NA"

                self.history[epoch]['val_loss'] = val_loss
                self.history[epoch]['val_metric'] = val_metric

                if self.early_stopping:
                    if val_loss < (self.best_val_loss - self.min_delta):
                        self.best_val_loss = val_loss
                        self.no_improvement_count = 0
                        self.best_weights = self.save_model_weights()
                    else:
                        self.no_improvement_count += 1
                        if self.no_improvement_count >= self.patience:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch}. Best validation loss: {self.best_val_loss:.4f}")
                            if self.best_weights:
                                self.restore_model_weights(self.best_weights)
                            break

                if self.lr_scheduler and self.no_improvement_count > 0 and self.no_improvement_count % 2 == 0:
                    old_lr = self.optimizer.learning_rate
                    self.optimizer.learning_rate *= self.lr_decay
                    if self.verbose:
                        print(f"Reducing learning rate from {old_lr:.6f} to {self.optimizer.learning_rate:.6f}")

                if self.verbose:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s} - val_loss: {val_loss:.4f} - val_{val_metric_s}")
            else:
                if self.verbose:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        if self.early_stopping and self.best_weights:
            self.restore_model_weights(self.best_weights)
            
        if val_dataset:
            self.calibrate_temperature(val_X, val_y)
            
        return self

    def predict(self, dataset):
        raw_preds = self.forward_propagation(dataset.X, training=False)
        
        if hasattr(self, 'temperature') and self.temperature != 1.0:
            logits = np.log(raw_preds / (1 - raw_preds + 1e-15))
            calibrated_preds = 1 / (1 + np.exp(-logits / self.temperature))
            return calibrated_preds
        
        return raw_preds

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")
        
    def save(self, file_path):
        model_data = {
            'layers': self.layers,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'metric': self.metric,
            'temperature': self.temperature if hasattr(self, 'temperature') else 1.0
        }
        np.savez(file_path, **model_data)

    def load(self, file_path):
        model_data = np.load(file_path, allow_pickle=True)
        self.layers = model_data['layers']
        self.optimizer = model_data['optimizer']
        self.loss = model_data['loss']
        self.metric = model_data['metric']
        self.temperature = model_data['temperature'] if 'temperature' in model_data else 1.0

