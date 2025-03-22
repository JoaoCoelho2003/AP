import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    
    ffn_output = Add()([attention_output, ffn_output])
    sequence_output = LayerNormalization(epsilon=1e-6)(ffn_output)
    
    return sequence_output

def build_transformer_model(vocab_size, embedding_dim, embedding_matrix, max_seq_length, 
                           head_size=64, num_heads=8, ff_dim=256, num_transformer_blocks=4, 
                           dropout_rate=0.3, l2_reg=0.001):
    inputs = Input(shape=(max_seq_length,))
    
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_seq_length,
        trainable=False
    )(inputs)
    
    x = Dropout(dropout_rate)(embedding_layer)
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)
    
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_transformer_model(X_train, y_train, X_val, y_val, vocab_size, embedding_dim, 
                           embedding_matrix, max_seq_length, 
                           model_path='../../trained_models/tensorflow/transformer_model.h5'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model = build_transformer_model(vocab_size, embedding_dim, embedding_matrix, max_seq_length)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    model = tf.keras.models.load_model(model_path)
    
    return model, history