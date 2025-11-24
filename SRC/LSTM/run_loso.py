# ======================================================
# SCRIPT 3: Ejecución LOSO
# Archivo: run_loso.py
# ======================================================
import os
from load_windows_loso import load_windows_by_subject, loso_split
from model_lstm_emotions import train_and_evaluate
WINDOWS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Ventanas")
)


# <<--- AQUÍ CAMBIAS EL SUJETO QUE QUEDA FUERA
SUBJECT_OUT = "19M"

subjects_data = load_windows_by_subject(WINDOWS_DIR)
print("Sujetos detectados:", list(subjects_data.keys()))
(
    X_train, y_train,
    X_val, y_val,
    X_test_int, y_test_int,
    X_test_ext, y_test_ext
) = loso_split(subjects_data, SUBJECT_OUT)

train_and_evaluate(
    X_train, y_train,
    X_val, y_val,
    X_test_int, y_test_int,
    X_test_ext, y_test_ext
)
