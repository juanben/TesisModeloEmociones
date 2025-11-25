# ======================================================
# model_lstm_attention.py
# Modelo LSTM bidireccional + Atención + Entrenamiento
# ======================================================

import tensorflow as tf
from tensorflow.keras import layers, regularizers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# ======================================================
# ATENCIÓN COMPATIBLE (Keras 3 + TF 2.16)
# ======================================================
def attention_block(lstm_outputs):
    """
    Atención tipo Bahdanau usando solo operaciones válidas de Keras.
    lstm_outputs: (batch, timesteps, features)
    """

    # --- Score ---
    score_first = layers.Dense(128, activation="tanh", name="attn_dense_1")(lstm_outputs)
    score = layers.Dense(1, name="attn_dense_2")(score_first)
    attention_weights = layers.Softmax(axis=1, name="attn_softmax")(score)

    # --- Weighted sum ---
    weighted = layers.Multiply(name="attn_multiply")([lstm_outputs, attention_weights])
    context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name="attn_reduce")(weighted)

    return context


# ======================================================
# CONSTRUCCIÓN DEL MODELO
# ======================================================
def build_lstm_model(
    input_shape,
    num_classes=4,
    lstm_units=64,
    dropout_rate=0.3,
    l2_reg=1e-4,
    learning_rate=1e-3,
):
    inputs = layers.Input(shape=input_shape, name="input_sequence")

    x = layers.BatchNormalization(name="bn_input")(inputs)

    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
        ),
        name="bilstm_1",
    )(x)

    x = layers.Dropout(dropout_rate, name="dropout_attention_1")(x)

    context = attention_block(x)

    x = layers.Dense(
        lstm_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_projection",
    )(context)

    x = layers.Dropout(dropout_rate, name="dropout_attention_2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_Attention_Emotion")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ======================================================
# CALLBACKS
# ======================================================
def get_callbacks(patience_es=20, patience_rlr=8):
    cb_early = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_es,
        restore_best_weights=True,
        verbose=1,
    )

    cb_rlr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience_rlr,
        min_lr=1e-6,
        verbose=1,
    )

    return [cb_early, cb_rlr]


# ======================================================
# ENTRENAR Y EVALUAR (compatibilidad con run_loso)
# ======================================================
def train_and_evaluate(
    X_train, y_train,
    X_val, y_val,
    X_test_int, y_test_int,
    X_test_ext, y_test_ext,
    epochs=100,
    batch_size=64
):

    print("\n==============================")
    print("  Entrenando modelo ATENCIÓN")
    print("==============================")

    input_shape = X_train.shape[1:]

    model = build_lstm_model(input_shape=input_shape, num_classes=4)
    cbs = get_callbacks()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1
    )

    # ===========================
    # TEST INTERNO
    # ===========================
    print("\n===== TEST INTERNO =====")
    y_pred_int = np.argmax(model.predict(X_test_int), axis=1)
    print(classification_report(y_test_int, y_pred_int, digits=3))
    print(confusion_matrix(y_test_int, y_pred_int))

    # ===========================
    # TEST EXTERNO
    # ===========================
    print("\n===== TEST EXTERNO LOSO =====")
    y_pred_ext = np.argmax(model.predict(X_test_ext), axis=1)
    print(classification_report(y_test_ext, y_pred_ext, digits=3))
    print(confusion_matrix(y_test_ext, y_pred_ext))

    return model, history
