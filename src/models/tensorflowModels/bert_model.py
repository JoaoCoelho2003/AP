import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_bert_model(dropout_rate=0.3, l2_reg=0.001):
    bert_preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )
    bert_encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        trainable=True
    )
    
    text_input = Input(shape=(), dtype=tf.string)
    
    preprocessed_text = bert_preprocessor(text_input)
    
    bert_outputs = bert_encoder(preprocessed_text)
    
    pooled_output = bert_outputs["pooled_output"]
    
    x = Dropout(dropout_rate)(pooled_output)
    
    x = Dense(256, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=text_input, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=2e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_bert_model(texts_train, y_train, texts_val, y_val, model_path='../../trained_models/tensorflow/bert_model.h5'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model = build_bert_model()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]
    
    history = model.fit(
        texts_train, y_train,
        validation_data=(texts_val, y_val),
        epochs=5,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
    
    return model, history