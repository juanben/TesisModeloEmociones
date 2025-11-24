# ======================================================
# SCRIPT 2: Modelo LSTM Multimodal + Entrenamiento + Evaluación
# Archivo: model_lstm_emotions.py
# ======================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix


def build_emotion_model(input_shape, num_classes):

    inputs = keras.Input(shape=input_shape)

    x = layers.BatchNormalization()(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate(X_train, y_train, X_val, y_val,
                        X_test_int, y_test_int,
                        X_test_ext, y_test_ext):

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = int(max(y_train.max(), y_val.max(), y_test_int.max(), y_test_ext.max()) + 1)

    model = build_emotion_model(input_shape, num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=callbacks
    )

    y_pred_int = np.argmax(model.predict(X_test_int), axis=1)
    print("\n===== TEST INTERNO =====")
    print(classification_report(y_test_int, y_pred_int))
    print(confusion_matrix(y_test_int, y_pred_int))

    y_pred_ext = np.argmax(model.predict(X_test_ext), axis=1)
    print("\n===== TEST EXTERNO LOSO =====")
    print(classification_report(y_test_ext, y_pred_ext))
    print(confusion_matrix(y_test_ext, y_pred_ext))

    return model
