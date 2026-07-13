# ======================================================
# SCRIPT 3: Ejecución LOSO
# Archivo: run_loso.py
# ======================================================
import os
import matplotlib.pyplot as plt
import numpy as np
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
# CONFIGURACIÓN DEL MODO DE ENTRENAMIENTO
# ======================================================
MODO_GLOBAL = True      # True: Entrena todos los sujetos uno por uno. False: Solo entrena el sujeto manual.
SUBJECT_OUT = "10H"     # Sujeto manual (solo se usa si MODO_GLOBAL = False)


# ======================================================
# CARGA DE VENTANAS Y PREPARACIÓN DE SUJETOS
# ======================================================
subjects_data = load_windows_by_subject(WINDOWS_DIR)
detected_subjects = list(subjects_data.keys())
print("Sujetos detectados:", detected_subjects)

# Determinar la lista de sujetos a procesar según la bandera
if MODO_GLOBAL:
    subjects_to_process = detected_subjects
    print(f"🌍 MODO GLOBAL ACTIVADO: Se entrenarán los {len(subjects_to_process)} sujetos secuencialmente.\n")
else:
    if SUBJECT_OUT not in subjects_data:
        raise ValueError(f"El sujeto manual '{SUBJECT_OUT}' no se encuentra en los datos detectados.")
    subjects_to_process = [SUBJECT_OUT]
    print(f"🕹 MODO MANUAL ACTIVADO: Solo se entrenará dejando fuera al sujeto: {SUBJECT_OUT}\n")


# ======================================================
# BUCLE DE ENTRENAMIENTO (LOSO)
# ======================================================
for current_subject in subjects_to_process:
    print(f"============ 🚀 Iniciando LOSO dejando fuera a: {current_subject} ============")
    
    (
        X_train, y_train,
        X_val, y_val,
        X_test_int, y_test_int,
        X_test_ext, y_test_ext,
        mean, std
    ) = loso_split(subjects_data, current_subject)

    # ======================================================
    # ENTRENAMIENTO Y EVALUACIÓN
    # ======================================================
    model, history = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test_int, y_test_int,
        X_test_ext, y_test_ext
    )
    
    # ======================================================
    # GUARDAR MODELO + SCALER
    # ======================================================
    os.makedirs("Modelo", exist_ok=True)
      
    model_path = f"Modelo/model_{SELECTED_MODEL}_LOSO_{current_subject}.keras"
    scaler_path = f"Modelo/scaler_{SELECTED_MODEL}_LOSO_{current_subject}.npz"

    model.save(model_path)
    np.savez(scaler_path, mean=mean, std=std)

    print(f"✔ Modelo guardado en: {model_path}")
    print(f"✔ Scaler (mean/std) guardado en: {scaler_path}")

    # ======================================================
    # GRAFICAR LEARNING CURVES
    # ======================================================
    os.makedirs("graficas", exist_ok=True)

    # Archivos según sujeto
    loss_path = f"graficas/loss_{current_subject}.png"
    acc_path  = f"graficas/accuracy_{current_subject}.png"

    # ======= LOSS =======
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss - LOSO {current_subject}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    print(f"✔ Gráfica guardada en: {loss_path}")

    # ======= ACCURACY =======
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training vs Validation Accuracy - LOSO {current_subject}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    print(f"✔ Gráfica guardada en: {acc_path}")
    print(f"============  Finalizado LOSO para sujeto {current_subject} ============\n")

print("🎉 Proceso completo terminado con éxito.")