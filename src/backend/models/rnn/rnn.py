from copy import deepcopy
from typing import Tuple
import numpy as np
from models.rnn.activation import TanhActivation, ActivationLayer
from models.rnn.layers import Layer

class RNN(Layer):
    def __init__(self, n_units: int, activation: ActivationLayer = None, bptt_trunc: int = 5, input_shape: Tuple = None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = TanhActivation() if activation is None else activation
        self.bptt_trunc = bptt_trunc
        self.W, self.V, self.U = None, None, None

    def initialize(self, optimizer):
        _, input_dim = self.input_shape
        limit = 1 / np.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / np.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.U_opt, self.V_opt, self.W_opt = deepcopy(optimizer), deepcopy(optimizer), deepcopy(optimizer)

    def forward_propagation(self, input: np.ndarray) -> np.ndarray:
        batch_size, timesteps, _ = input.shape
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, self.n_units))

        for t in range(timesteps):
            self.state_input[:, t] = input[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            self.states[:, t] = self.activation.activation_function(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward_propagation(self, accum_grad: np.ndarray, input: np.ndarray) -> np.ndarray:
        _, timesteps, _ = accum_grad.shape
        grad_U, grad_V, grad_W = np.zeros_like(self.U), np.zeros_like(self.V), np.zeros_like(self.W)
        accum_grad_next = np.zeros_like(accum_grad)

        max_norm = 5.0

        for t in reversed(range(timesteps)):
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.derivative(self.state_input[:, t])
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)

            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t + 1)):
                grad_U += grad_wrt_state.T.dot(input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.derivative(self.state_input[:, t_ - 1])

        grad_U = np.clip(grad_U, -max_norm, max_norm)
        grad_V = np.clip(grad_V, -max_norm, max_norm)
        grad_W = np.clip(grad_W, -max_norm, max_norm)

        self.U[:] = self.U_opt.update(self.U, grad_U)
        self.V[:] = self.V_opt.update(self.V, grad_V)
        self.W[:] = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 32, epochs: int = 10):
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, num_samples, batch_size):
                batch_X = X_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]

                predictions = self.forward_propagation(batch_X)

                accum_grad = predictions - batch_y

                self.backward_propagation(accum_grad, batch_X)

            print(f"Epoch {epoch + 1}/{epochs} completed.")

    def output_shape(self) -> tuple:
        return self.input_shape

    def parameters(self) -> int:
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))

        predictions = self.forward_propagation(X)
        return predictions[:, -1, :]


    def save(self, file_path: str):
        np.savez(file_path, U=self.U, V=self.V, W=self.W)

    def load(self, file_path: str):
        data = np.load(file_path)
        self.U, self.V, self.W = data['U'], data['V'], data['W']
