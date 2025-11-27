import os
import numpy as np
import pandas as pd

INPUT_DIR = "DatasetLimpio"
OUTPUT_DIR = "Ventanas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SECONDS = 3
OVERLAP_RATIO = 0.5


# ==========================================================
# 1) CALCULAR MEDIA Y STD GLOBAL DE ECG (ecg_smooth)
# ==========================================================
print("📌 Calculando media y desviación global del ECG (ecg_smooth)...\n")

all_ecg = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    if "ecg_smooth" not in df.columns:
        print(f"⚠️ Advertencia: archivo sin ecg_smooth -> {file}")
        continue

    all_ecg.append(df["ecg_smooth"].values)

# concatenar todo
all_ecg = np.concatenate(all_ecg)

GLOBAL_MEAN = all_ecg.mean()
GLOBAL_STD  = all_ecg.std()
if GLOBAL_STD == 0:
    GLOBAL_STD = 1.0

print(f"   ✔ Global Mean ECG: {GLOBAL_MEAN:.4f}")
print(f"   ✔ Global Std  ECG: {GLOBAL_STD:.4f}\n")


# ==========================================================
# 2) GENERAR VENTANAS
# ==========================================================
def generar_ventanas(df, fps, window_seconds, overlap_ratio):

    samples_per_window = int(window_seconds * fps)
    step = int(samples_per_window * (1 - overlap_ratio))

    features_cols = []

    # ---------
    # ECG BASE
    # ---------
    features_cols.append("ecg_norm")     # canal 0
    features_cols.append("ecg_diff")
    features_cols.append("ecg_speed")
    features_cols.append("ecg_energy")

    # ---------
    # POSE 33x3
    # ---------
    for i in range(33):
        features_cols.append(f"{i}_x")
        features_cols.append(f"{i}_y")
        features_cols.append(f"{i}_z")

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

        # emoción dominante
        unique, counts = np.unique(win_labels, return_counts=True)
        dominant = unique[np.argmax(counts)]

        X_windows.append(window)
        y_windows.append(dominant)

    return np.array(X_windows), np.array(y_windows)


# ==========================================================
# 3) PROCESAR PARTICIPANTE
# ==========================================================
def procesar_archivo(path):
    subject_id = os.path.basename(path).split("_")[0]

    print(f"\n🔵 Procesando: {subject_id}")

    df = pd.read_csv(path)

    # ============================
    # A) NORMALIZACIÓN GLOBAL ECG
    # ============================
    df["ecg_norm"] = (df["ecg_smooth"] - GLOBAL_MEAN) / GLOBAL_STD

    # ============================
    # B) DERIVADAS DEL ECG NORMALIZADO
    # ============================
    df["ecg_diff"]  = df["ecg_norm"].diff().fillna(0)
    df["ecg_speed"] = df["ecg_norm"].diff().abs().fillna(0)
    df["ecg_energy"] = df["ecg_norm"] ** 2

    # ============================
    # C) GENERAR VENTANAS
    # ============================
    fps = 10.4  # tu tasa fija
    X, y = generar_ventanas(df, fps,
                            window_seconds=WINDOW_SECONDS,
                            overlap_ratio=OVERLAP_RATIO)

    # guardar
    np.save(os.path.join(OUTPUT_DIR, f"{subject_id}_windows.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"{subject_id}_labels.npy"), y)

    print(f"   ✔ Ventanas generadas: {X.shape[0]}")
    print(f"   ✔ Shape ventana: {X.shape[1:]} (samples, features)")


# ==========================================================
# 4) LOOP PRINCIPAL
# ==========================================================
if __name__ == "__main__":

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    for f in files:
        procesar_archivo(os.path.join(INPUT_DIR, f))

    print("\n🎉 PROCESO COMPLETADO — VENTANAS GENERADAS")
