import os
import re
import numpy as np
import pandas as pd

INPUT_DIR = "DatasetLimpio"
OUTPUT_DIR = "Ventanas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SECONDS = 3          # Sección comun en todos los trabajos relacionados
OVERLAP_RATIO = 0.5         # 50%
FEATURE_COUNT = 100         # 1 ECG + 99 pose

def cargar_dataframe(path):
    df = pd.read_csv(path)
    # Asegurar tiempo relativo
    if "time_sec" not in df.columns:
        df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df

def calcular_fps(df):
    diffs = df["time_sec"].diff().dropna()
    return 1.0 / diffs.mean()

def generar_ventanas(df, fps, window_seconds=5, overlap_ratio=0.5):

    samples_per_window = int(window_seconds * fps)
    step = int(samples_per_window * (1 - overlap_ratio))

    features_cols = []

    # ECG primero
    features_cols.append("ecg_smooth")

    # Luego los 33*3 landmarks
    for i in range(33):
        features_cols.append(f"{i}_x")
        features_cols.append(f"{i}_y")
        features_cols.append(f"{i}_z")

    # Extraer arrays
    data = df[features_cols].values
    labels = df["label"].values

    X_windows = []
    y_windows = []

    for start in range(0, len(df) - samples_per_window, step):
        end = start + samples_per_window
        window_data = data[start:end]

        window_labels = labels[start:end]

            # ✅ DESCARTAR VENTANAS INCOMPLETAS
        if window_data.shape[0] != samples_per_window:
            continue

        # Asignar emoción dominante
        unique, counts = np.unique(window_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]

        X_windows.append(window_data)
        y_windows.append(dominant_label)

    return np.array(X_windows), np.array(y_windows)

def procesar_participante(file_path):
    subject_id = os.path.basename(file_path).split("_")[0]

    print(f"Procesando participante: {subject_id}")

    df = cargar_dataframe(file_path)
    fps = 10.4

    X, y = generar_ventanas(df, fps,
                            window_seconds=WINDOW_SECONDS,
                            overlap_ratio=OVERLAP_RATIO)

    # Guardar
    np.save(os.path.join(OUTPUT_DIR, f"{subject_id}_windows.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"{subject_id}_labels.npy"), y)

    print(f"  ✔ Ventanas: {X.shape[0]}")
    print(f"  ✔ Shape ventana: {X.shape[1:]} (samples, features)")
    print(f"  ✔ Guardado en carpeta Ventanas\n")


if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    for f in files:
        procesar_participante(os.path.join(INPUT_DIR, f))

    print("\n🎉 PROCESO COMPLETADO — VENTANAS GENERADAS\n")
