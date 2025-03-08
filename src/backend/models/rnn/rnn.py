from copy import deepcopy
from typing import Tuple
import numpy as np
from models.rnn.activation import TanhActivation, ActivationLayer
from models.rnn.layers import Layer
from models.rnn.optimizers import SGD

class RNN(Layer):
    def __init__(self, n_units: int, activation: ActivationLayer = None, bptt_trunc: int = 5, input_shape: Tuple = None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = TanhActivation() if activation is None else activation
        self.bptt_trunc = bptt_trunc
        self.W, self.V, self.U = None, None, None

    def initialize(self, optimizer):
        timesteps, input_dim = self.input_shape
        limit = 1 / np.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / np.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.U_opt, self.V_opt, self.W_opt = deepcopy(optimizer), deepcopy(optimizer), deepcopy(optimizer)

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, timesteps, input_dim = input.shape
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, self.n_units))
        for t in range(timesteps):
            self.state_input[:, t] = input[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            self.states[:, t] = self.activation.activation_function(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)
        return self.outputs

    def backward_propagation(self, accum_grad: np.ndarray) -> np.ndarray:
        _, timesteps, _ = accum_grad.shape
        grad_U, grad_V, grad_W = np.zeros_like(self.U), np.zeros_like(self.V), np.zeros_like(self.W)
        accum_grad_next = np.zeros_like(accum_grad)
        for t in reversed(range(timesteps)):
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.derivative(self.state_input[:, t])
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t + 1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.derivative(self.state_input[:, t_ - 1])
        self.U, self.V, self.W = self.U_opt.update(self.U, grad_U), self.V_opt.update(self.V, grad_V), self.W_opt.update(self.W, grad_W)
        return accum_grad_next

    def output_shape(self) -> tuple:
        return self.input_shape

    def parameters(self) -> int:
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)

    def save(self, file_path: str):
        np.savez(file_path, U=self.U, V=self.V, W=self.W)

    def load(self, file_path: str):
        data = np.load(file_path)
        self.U, self.V, self.W = data['U'], data['V'], data['W']