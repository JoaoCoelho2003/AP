from copy import deepcopy
from typing import Tuple
import numpy as np
from models.rnn.activation import TanhActivation, SigmoidActivation, ActivationLayer
from models.rnn.layers import Layer
from models.rnn.losses import BinaryCrossEntropy

class RNN(Layer):
    def __init__(self, n_units: int, activation: ActivationLayer = None, bptt_trunc: int = 5, 
                 input_shape: Tuple = None, embedding_matrix: np.ndarray = None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = TanhActivation() if activation is None else activation
        self.output_activation = SigmoidActivation()
        self.bptt_trunc = bptt_trunc
        self.W, self.V, self.U = None, None, None
        self.loss_function = BinaryCrossEntropy()
        
        self.embedding_matrix = embedding_matrix
        self.use_embeddings = embedding_matrix is not None
        self.embedding_dim = embedding_matrix.shape[1] if self.use_embeddings else None
        
        self.l2_reg = 0.001
        
        self.dropout_rate = 0.5
        self.use_dropout = True
        
        self.max_norm = 3.0

    def initialize(self, optimizer):
        if self.use_embeddings:
            input_dim = self.embedding_dim
        else:
            _, input_dim = self.input_shape
        
        limit_u = np.sqrt(2 / (input_dim + self.n_units))
        self.U = np.random.uniform(-limit_u, limit_u, (self.n_units, input_dim))
        
        limit_w = np.sqrt(2 / (self.n_units + self.n_units))
        self.W = np.random.uniform(-limit_w, limit_w, (self.n_units, self.n_units))
        
        limit_v = np.sqrt(2 / (self.n_units + 1))
        self.V = np.random.uniform(-limit_v, limit_v, (1, self.n_units))
        
        self.U_opt, self.V_opt, self.W_opt = deepcopy(optimizer), deepcopy(optimizer), deepcopy(optimizer)

        self.dropout_rate = 0.3
        self.use_dropout = True

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        if self.use_embeddings:
            batch_size, timesteps = input.shape
            embedded_input = np.zeros((batch_size, timesteps, self.embedding_dim))
            
            for i in range(batch_size):
                for j in range(timesteps):
                    idx = input[i, j]
                    if idx < len(self.embedding_matrix):
                        embedded_input[i, j] = self.embedding_matrix[idx]
            
            batch_size, timesteps, _ = embedded_input.shape
            self.embedded_input = embedded_input
        else:
            batch_size, timesteps, _ = input.shape
        
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, 1))
        
        if training and self.use_dropout:
            self.dropout_masks = []

        for t in range(timesteps):
            if self.use_embeddings:
                self.state_input[:, t] = self.embedded_input[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            else:
                self.state_input[:, t] = input[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            
            self.states[:, t] = self.activation.activation_function(self.state_input[:, t])
            
            if training and self.use_dropout:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.states[:, t].shape) / (1 - self.dropout_rate)
                self.dropout_masks.append(dropout_mask)
                self.states[:, t] *= dropout_mask
            
            self.outputs[:, t] = self.output_activation.activation_function(self.states[:, t].dot(self.V.T))

        return self.outputs

    def backward_propagation(self, accum_grad: np.ndarray, input: np.ndarray = None) -> np.ndarray:
        batch_size, timesteps, _ = accum_grad.shape
        
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        
        if self.use_embeddings:
            accum_grad_next = np.zeros_like(self.embedded_input)
        else:
            accum_grad_next = np.zeros_like(input)
        
        max_norm = 5.0

        for t in reversed(range(timesteps)):
            sigmoid_grad = self.output_activation.derivative(self.states[:, t].dot(self.V.T)) * accum_grad[:, t]
            
            grad_V += sigmoid_grad.T.dot(self.states[:, t])
            
            grad_wrt_state = sigmoid_grad.dot(self.V) * self.activation.derivative(self.state_input[:, t])
            
            if self.use_embeddings:
                for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t + 1)):
                    grad_U += grad_wrt_state.T.dot(self.embedded_input[:, t_])
                    accum_grad_next[:, t_] = grad_wrt_state.dot(self.U)
                    grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                    
                    if t_ > 0:
                        grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.derivative(self.state_input[:, t_ - 1])
            else:
                accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
                
                for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t + 1)):
                    grad_U += grad_wrt_state.T.dot(input[:, t_])
                    grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                    
                    if t_ > 0:
                        grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.derivative(self.state_input[:, t_ - 1])

        grad_U /= batch_size
        grad_V /= batch_size
        grad_W /= batch_size
        
        grad_U = np.clip(grad_U, -max_norm, max_norm)
        grad_V = np.clip(grad_V, -max_norm, max_norm)
        grad_W = np.clip(grad_W, -max_norm, max_norm)

        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 32, epochs: int = 10):
        num_samples = X_train.shape[0]
        self.epochs = epochs
    
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
                
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_X = X_train_shuffled[i:end_idx]
                batch_y = y_train_shuffled[i:end_idx]

                predictions = self.forward_propagation(batch_X, training=True)
                
                final_predictions = predictions[:, -1, :]
                
                batch_loss = self.loss_function.compute(batch_y, final_predictions)
                epoch_loss += batch_loss * batch_y.shape[0]
                
                predicted_classes = (final_predictions > 0.5).astype(int)
                correct_predictions += np.sum(predicted_classes == batch_y.reshape(-1, 1))
                total_predictions += batch_y.shape[0]
                
                accum_grad = np.zeros_like(predictions)
                
                output_grad = self.loss_function.derivative(batch_y, final_predictions)
                accum_grad[:, -1, :] = output_grad
                
                if self.use_embeddings:
                    self.backward_propagation(accum_grad)
                else:
                    self.backward_propagation(accum_grad, batch_X)

            train_accuracy = correct_predictions / total_predictions
            avg_epoch_loss = epoch_loss / num_samples

            val_predictions = self.forward_propagation(X_val, training=False)
            final_val_predictions = val_predictions[:, -1, :]
            val_loss = self.loss_function.compute(y_val, final_val_predictions)
            val_predicted_classes = (final_val_predictions > 0.5).astype(int)
            val_accuracy = np.sum(val_predicted_classes == y_val.reshape(-1, 1)) / y_val.shape[0]

            print(f"Epoch {epoch + 1}/{self.epochs} - loss: {avg_epoch_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
                
            if epoch > 0 and epoch % 3 == 0:
                new_lr = self.U_opt.lr * 0.8
                self.U_opt.lr = new_lr
                self.V_opt.lr = new_lr
                self.W_opt.lr = new_lr
                print(f"Reducing learning rate to {new_lr:.6f}")

    def output_shape(self) -> tuple:
        return self.input_shape

    def parameters(self) -> int:
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.use_embeddings and X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))

        predictions = self.forward_propagation(X, training=False)
        return (predictions[:, -1, :] > 0.5).astype(int)

    def save(self, file_path: str):
        if self.use_embeddings:
            np.savez(file_path, U=self.U, V=self.V, W=self.W, use_embeddings=np.array([True]))
        else:
            np.savez(file_path, U=self.U, V=self.V, W=self.W, use_embeddings=np.array([False]))

    def load(self, file_path: str):
        data = np.load(file_path)
        self.U, self.V, self.W = data['U'], data['V'], data['W']
        if 'use_embeddings' in data:
            self.use_embeddings = bool(data['use_embeddings'][0])

    def set_dropout(self, rate=0.3, use_dropout=True):
        self.dropout_rate = rate
        self.use_dropout = use_dropout