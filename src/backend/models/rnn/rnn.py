import numpy as np
from copy import deepcopy
from typing import Tuple
from models.rnn.layers import Layer
from models.rnn.loss import BinaryCrossEntropy

class RNN(Layer):
    def __init__(self, embedding_matrix, hidden_size=64, dropout_rate=0.2, l2_reg=0.001):
        self.embedding_matrix = embedding_matrix
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.loss_function = BinaryCrossEntropy()
        
        self.W = None
        self.U = None
        self.V = None
        self.b_h = None
        self.b_v = None
    
    def initialize(self, optimizer):
        limit_w = np.sqrt(1.0 / self.hidden_size)
        self.W = np.random.uniform(-limit_w, limit_w, (self.hidden_size, self.hidden_size))
        
        W_init = np.random.randn(self.hidden_size, self.hidden_size)
        u, _, v = np.linalg.svd(W_init, full_matrices=False)
        self.W = u @ v
        self.W *= 0.9
        
        limit_u = np.sqrt(1.0 / (self.embedding_dim + self.hidden_size))
        self.U = np.random.uniform(-limit_u, limit_u, (self.hidden_size, self.embedding_dim))
        
        limit_v = np.sqrt(1.0 / (1 + self.hidden_size))
        self.V = np.random.uniform(-limit_v, limit_v, (1, self.hidden_size))
        
        self.b_h = np.zeros((self.hidden_size, 1))
        self.b_v = np.zeros((1, 1))
        
        self.W_opt = deepcopy(optimizer)
        self.U_opt = deepcopy(optimizer)
        self.V_opt = deepcopy(optimizer)
        self.b_h_opt = deepcopy(optimizer)
        self.b_v_opt = deepcopy(optimizer)
        
        print(f"Initialized EmbeddingRNN with hidden size: {self.hidden_size}")
        print(f"Embedding matrix shape: {self.embedding_matrix.shape}")
    
    def _apply_dropout(self, x, training=True):
        if self.dropout_rate == 0 or not training:
            return x
        
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
        return x * mask
    
    def _layer_normalize(self, x, epsilon=1e-8):
        mean = np.mean(x, axis=1, keepdims=True)
        variance = np.var(x, axis=1, keepdims=True)
        return (x - mean) / np.sqrt(variance + epsilon)
    
    def forward_propagation(self, input_sequences, training=True):
        batch_size, seq_length = input_sequences.shape
        
        self.input_sequences = input_sequences
        
        self.embedded_input = np.zeros((batch_size, seq_length, self.embedding_dim))
        for i in range(batch_size):
            for t in range(seq_length):
                word_idx = input_sequences[i, t]
                if word_idx < self.vocab_size:
                    self.embedded_input[i, t] = self.embedding_matrix[word_idx]
        
        self.h = np.zeros((batch_size, seq_length + 1, self.hidden_size))
        self.state_input = np.zeros((batch_size, seq_length, self.hidden_size))
        
        for t in range(seq_length):
            x_t = self.embedded_input[:, t, :]
            
            if training and self.dropout_rate > 0:
                x_t = self._apply_dropout(x_t, training)
            
            self.state_input[:, t] = x_t.dot(self.U.T) + self.h[:, t-1].dot(self.W.T) + self.b_h.T
            self.h[:, t] = np.tanh(self.state_input[:, t])
            
            if training and self.dropout_rate > 0:
                self.h[:, t] = self._apply_dropout(self.h[:, t], training)
            
            self.h[:, t] = self._layer_normalize(self.h[:, t])
        
        final_h = self.h[:, seq_length-1]
        output = final_h.dot(self.V.T) + self.b_v.T
        self.output = 1 / (1 + np.exp(-np.clip(output, -5, 5)))
        
        return self.output
    
    def backward_propagation(self, y_true):
        batch_size, seq_length = self.input_sequences.shape
        
        output_grad = self.loss_function.derivative(y_true, self.output)
        
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        dV = np.zeros_like(self.V)
        db_h = np.zeros_like(self.b_h)
        db_v = np.zeros_like(self.b_v)
        
        dV = output_grad.T.dot(self.h[:, seq_length-1])
        db_v = np.sum(output_grad, axis=0, keepdims=True).T
        
        dh = output_grad.dot(self.V)
        
        for t in reversed(range(seq_length)):
            dh = self._layer_normalize(dh)
            
            dtanh = dh * (1 - np.tanh(self.state_input[:, t])**2)
            
            dU += dtanh.T.dot(self.embedded_input[:, t])
            dW += dtanh.T.dot(self.h[:, t-1])
            db_h += np.sum(dtanh, axis=0, keepdims=True).T
            
            if t > 0:
                dh = dtanh.dot(self.W)
        
        if self.l2_reg > 0:
            dW += self.l2_reg * self.W
            dU += self.l2_reg * self.U
            dV += self.l2_reg * self.V
        
        max_norm = 5.0
        for grad in [dW, dU, dV, db_h, db_v]:
            np.clip(grad, -max_norm, max_norm, out=grad)
        
        self.W = self.W_opt.update(self.W, dW)
        self.U = self.U_opt.update(self.U, dU)
        self.V = self.V_opt.update(self.V, dV)
        self.b_h = self.b_h_opt.update(self.b_h, db_h)
        self.b_v = self.b_v_opt.update(self.b_v, db_v)
        
        return None
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
        num_samples = X_train.shape[0]
        
        initial_lr = self.W_opt.lr
        best_val_metric = 0
        patience_counter = 0
        patience = 3
        
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X_train[indices], y_train[indices]
            
            total_loss, total_metric = 0, 0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_X = X_shuffled[i:end_idx]
                batch_y = y_shuffled[i:end_idx].reshape(-1, 1)
                
                predictions = self.forward_propagation(batch_X)
                
                loss = self.loss_function.compute(batch_y, predictions)
                
                if self.l2_reg > 0:
                    l2_loss = self.l2_reg * 0.5 * (
                        np.sum(self.W**2) + np.sum(self.U**2) + np.sum(self.V**2)
                    )
                    loss += l2_loss
                
                metric = np.mean(((predictions > 0.5).astype(int) == batch_y).astype(float))
                
                total_loss += loss
                total_metric += metric
                
                self.backward_propagation(batch_y)
                num_batches += 1
                
                if i % (5 * batch_size) == 0:
                    print(f"Epoch {epoch+1}, Batch {i//batch_size} - Loss: {loss:.4f}, Accuracy: {metric:.4f}")
            
            avg_loss = total_loss / num_batches
            avg_metric = total_metric / num_batches
            
            val_predictions = self.forward_propagation(X_val, training=False)
            val_loss = self.loss_function.compute(y_val.reshape(-1, 1), val_predictions)
            val_metric = np.mean(((val_predictions > 0.5).astype(int) == y_val.reshape(-1, 1)).astype(float))
            
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - metric: {avg_metric:.4f} - val_loss: {val_loss:.4f} - val_metric: {val_metric:.4f}")
            
            current_lr = initial_lr / (1 + epoch * 0.1)
            self.W_opt.lr = current_lr
            self.U_opt.lr = current_lr
            self.V_opt.lr = current_lr
            self.b_h_opt.lr = current_lr
            self.b_v_opt.lr = current_lr
            
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
                self.save("trained_models/embedding_rnn_best.npz")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    self.load("trained_models/embedding_rnn_best.npz")
                    break
    
    def output_shape(self):
        return (None, 1)
    
    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape) + np.prod(self.b_h.shape) + np.prod(self.b_v.shape)
    
    def predict(self, X):
        return self.forward_propagation(X, training=False)
    
    def save(self, file_path):
        np.savez(file_path, W=self.W, U=self.U, V=self.V, b_h=self.b_h, b_v=self.b_v)
    
    def load(self, file_path):
        data = np.load(file_path)
        self.W, self.U, self.V = data['W'], data['U'], data['V']
        self.b_h, self.b_v = data['b_h'], data['b_v']

