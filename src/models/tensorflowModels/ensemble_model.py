import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_ensemble_model(models, dropout_rate=0.3, l2_reg=0.001):
    inputs = []
    outputs = []
    
    for i, model in enumerate(models):
        input_layer = Input(shape=model.input_shape[1:], name=f"input_{i}")
        inputs.append(input_layer)
        
        model_copy = tf.keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        
        output = model_copy(input_layer)
        outputs.append(output)
    
    if len(outputs) > 1:
        concatenated = Concatenate()(outputs)
    else:
        concatenated = outputs[0]
    
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(concatenated)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    final_output = Dense(1, activation="sigmoid")(x)
    
    ensemble_model = Model(inputs=inputs, outputs=final_output)
    
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return ensemble_model

def train_ensemble_model(X_trains, y_train, X_vals, y_val, models, model_path='../../trained_models/tensorflow/ensemble_model.h5'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    ensemble_model = build_ensemble_model(models)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    history = ensemble_model.fit(
        X_trains, y_train,
        validation_data=(X_vals, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    ensemble_model = tf.keras.models.load_model(model_path)
    
    return ensemble_model, history