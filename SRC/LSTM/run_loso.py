# ======================================================
# SCRIPT 3: Ejecución LOSO
# Archivo: run_loso.py
# ======================================================
import os
from load_windows_loso import load_windows_by_subject, loso_split

WINDOWS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Ventanas")
)

# ======================================================
# SELECCIÓN DEL MODELO
# ======================================================
# OPCIONES:
#   "clasico"
#   "attention"
SELECTED_MODEL = "attention"   # <-- cambia aquí

if SELECTED_MODEL == "clasico":
    from model_lstm_emotions import train_and_evaluate
    print("🔵 Usando modelo LSTM clásico")
elif SELECTED_MODEL == "attention":
    from model_lstm_atention import train_and_evaluate
    print("🟣 Usando modelo LSTM + Atención")
else:
    raise ValueError("Modelo no reconocido. Usa 'clasico' o 'attention'.")


# ======================================================
# SUJETO A DEJAR FUERA (LOSO)
# ======================================================
SUBJECT_OUT = "10H"   # <-- cámbialo para seleccionar qué sujeto queda fuera


# ======================================================
# CARGA DE VENTANAS
# ======================================================
subjects_data = load_windows_by_subject(WINDOWS_DIR)
print("Sujetos detectados:", list(subjects_data.keys()))

(
    X_train, y_train,
    X_val, y_val,
    X_test_int, y_test_int,
    X_test_ext, y_test_ext
) = loso_split(subjects_data, SUBJECT_OUT)


# ======================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ======================================================
train_and_evaluate(
    X_train, y_train,
    X_val, y_val,
    X_test_int, y_test_int,
    X_test_ext, y_test_ext
)
