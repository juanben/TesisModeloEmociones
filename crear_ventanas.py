import os
import numpy as np
import pandas as pd

INPUT_DIR = "DatasetLimpio"
OUTPUT_DIR = "Ventanas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SECONDS = 3
OVERLAP_RATIO = 0.75  # Aumentado al 75% para maximizar el soporte de Miedo/Alegría/Ira

def generar_ventanas(df, fps, window_seconds, overlap_ratio):
    samples_per_window = int(window_seconds * fps)
    step = int(samples_per_window * (1 - overlap_ratio))

    features_cols = []
    # ---------
    # ECG BASE (4 Canales Fisiológicos Avanzados)
    # ---------
    features_cols.append("ecg_norm")     
    features_cols.append("ecg_diff")
    features_cols.append("ecg_speed")
    features_cols.append("ecg_energy")  # ¡Ahora sí garantizado!

    # ---------
    # POSE OPTIMIZADA (Tronco superior, coordenadas estables X e Y únicamente)
    # ---------
    for i in range(17):  # Landmarks del 0 al 16 (rostro, hombros, codos, manos)
        features_cols.append(f"{i}_x")
        features_cols.append(f"{i}_y")

    data = df[features_cols].values
    labels = df["label"].values

    X_windows = []
    y_windows = []

    for start in range(0, len(df) - samples_per_window, step):
        end = start + samples_per_window
        window = data[start:end]

        if window.shape[0] != samples_per_window:
            continue

        win_labels = labels[start:end]
        unique, counts = np.unique(win_labels, return_counts=True)
        dominant = unique[np.argmax(counts)]

        X_windows.append(window)
        y_windows.append(dominant)

    return np.array(X_windows), np.array(y_windows)

def procesar_archivo(path):
    subject_id = os.path.basename(path).split("_")[0]
    print(f"\n🔵 Procesando e Normalizando Sujeto: {subject_id}")

    df = pd.read_csv(path)

    # CORRECCIÓN CRÍTICA: Normalización Z-Score local (Evita Data Leakage inter-sujeto)
    sub_mean = df["ecg_smooth"].mean()
    sub_std = df["ecg_smooth"].std() if df["ecg_smooth"].std() != 0 else 1.0

    df["ecg_norm"] = (df["ecg_smooth"] - sub_mean) / sub_std
    
    # CORRECCIÓN DE ORDEN: Calcular variables ANTES de llamar a generar_ventanas
    df["ecg_diff"]  = df["ecg_norm"].diff().fillna(0)
    df["ecg_speed"] = df["ecg_norm"].diff().abs().fillna(0)
    df["ecg_energy"] = df["ecg_norm"] ** 2

    LABEL_MAP = {"neutro": 0, "miedo": 1, "ira": 2, "alegria": 3}
    df["label"] = df["label"].map(LABEL_MAP)

    fps = 10.4  
    X, y = generar_ventanas(df, fps, window_seconds=WINDOW_SECONDS, overlap_ratio=OVERLAP_RATIO)

    np.save(os.path.join(OUTPUT_DIR, f"{subject_id}_windows.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"{subject_id}_labels.npy"), y)

    print(f"   ✔ Ventanas generadas exitosamente: {X.shape[0]}")
    print(f"   ✔ Dimensiones de entrada seguras: {X.shape[1:]} (pasos de tiempo, variables)")

if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    for f in files:
        procesar_archivo(os.path.join(INPUT_DIR, f))

    print("\n🎉 PROCESO COMPLETADO — DATASET LISTO PARA EL EVALUADOR LOSO")