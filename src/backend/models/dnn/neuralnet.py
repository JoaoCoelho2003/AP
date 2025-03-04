#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from models.dnn.layers import DenseLayer
from models.dnn.losses import LossFunction, MeanSquaredError
from models.dnn.optimizer import Optimizer
from models.dnn.metrics import mse


class NeuralNetwork:
 
    def __init__(self, epochs = 100, batch_size = 128, optimizer = None,
                 learning_rate = 0.01, momentum = 0.90, verbose = False, 
                 loss = MeanSquaredError,
                 metric:callable = mse):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum= momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y = None,shuffle = True):
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

    def fit(self, dataset, val_dataset=None):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            # Store mini-batch data for epoch loss and quality metrics calculation
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                # Forward propagation
                output = self.forward_propagation(X_batch, training=True)
                # Backward propagation
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # Compute training loss
            loss = self.loss.loss(y_all, output_x_all)

            # Compute training metric
            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            # Save training loss and metric
            self.history[epoch] = {'loss': loss, 'metric': metric}

            # Validation Step
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

                # Store validation metrics
                self.history[epoch]['val_loss'] = val_loss
                self.history[epoch]['val_metric'] = val_metric

                # Print training + validation metrics
                if self.verbose:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s} - val_loss: {val_loss:.4f} - val_{val_metric_s}")
            else:
                if self.verbose:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self


    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

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
            'metric': self.metric
        }
        np.savez(file_path, **model_data)

    def load(self, file_path):
        model_data = np.load(file_path, allow_pickle=True)
        self.layers = model_data['layers']
        self.optimizer = model_data['optimizer']
        self.loss = model_data['loss']
        self.metric = model_data['metric']