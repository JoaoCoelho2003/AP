import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, SpatialDropout1D, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_lstm_model(vocab_size, embedding_dim, embedding_matrix, max_seq_length, 
                     dropout_rate=0.3, recurrent_dropout=0.3, l2_reg=0.001):
    model = Sequential([
        Input(shape=(max_seq_length,)),
        
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_length,
            trainable=False
        ),
        
        SpatialDropout1D(dropout_rate),
        
        Bidirectional(LSTM(128, return_sequences=True, dropout=dropout_rate, 
                          recurrent_dropout=recurrent_dropout)),
        
        Bidirectional(LSTM(64, dropout=dropout_rate, recurrent_dropout=recurrent_dropout)),
        
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def build_gru_model(vocab_size, embedding_dim, embedding_matrix, max_seq_length, 
                   dropout_rate=0.3, recurrent_dropout=0.3, l2_reg=0.001):
    model = Sequential([
        Input(shape=(max_seq_length,)),
        
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_length,
            trainable=False
        ),
        
        SpatialDropout1D(dropout_rate),
        
        Conv1D(128, 5, activation='relu', padding='same'),
        
        Bidirectional(GRU(128, return_sequences=True, dropout=dropout_rate, 
                         recurrent_dropout=recurrent_dropout)),
        
        Bidirectional(GRU(64, dropout=dropout_rate, recurrent_dropout=recurrent_dropout)),
        
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_rnn_model(X_train, y_train, X_val, y_val, vocab_size, embedding_dim, 
                   embedding_matrix, max_seq_length, model_type='lstm', 
                   model_path='../../trained_models/tensorflow/rnn_model.h5'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if model_type == 'lstm':
        model = build_lstm_model(vocab_size, embedding_dim, embedding_matrix, max_seq_length)
    else:
        model = build_gru_model(vocab_size, embedding_dim, embedding_matrix, max_seq_length)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    model = tf.keras.models.load_model(model_path)
    
    return model, history