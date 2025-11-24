import pandas as pd
import numpy as np
import glob
import os
import re

OUTPUT_DIR = "DatasetLimpio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in glob.glob("datasets_etiquetados/*.csv")]

# ------------------------------
# Detectar tramos planos
# ------------------------------
def detectar_ecg_plano(df, col="ecg_smooth", ventana=25, tol=3):

    valores = df[col].values
    n = len(valores)

    if n == 0 or n < ventana:
        return np.zeros(n, dtype=bool)

    dif = np.abs(np.diff(valores))
    plano = (dif < tol).astype(int)

    conv = np.convolve(plano, np.ones(ventana), mode="same")
    conv = conv >= ventana

    conv_full = np.zeros(n, dtype=bool)
    conv_full[:len(conv)] = conv

    return conv_full


for file in files:
    print(f"Procesando: {file}")

    df = pd.read_csv(file)

    # =============================
    # PRESERVAR ETIQUETAS Y TIEMPO
    # =============================
    label_backup = df["label"].copy() if "label" in df.columns else None
    time_backup  = df["time_sec"].copy() if "time_sec" in df.columns else None

    # =============================
    # COLUMNAS DE POSE
    # =============================
    cols_pose = [c for c in df.columns if re.match(r"\d+_[xyz]$", c)]

    # ✅ NUEVO: eliminar filas donde TODOS los landmarks == 0
    df = df[(df[cols_pose].abs().sum(axis=1) != 0)]

    # 1) eliminar NaNs de pose
    df = df.dropna(subset=cols_pose)

    # 2) Filtrar ECG sin afectar emoción
    df = df[(df["ecg"] > 100) & (df["ecg"] < 4095)]

    # 3) Mantener frames si 80% landmarks están en rango
    mask = ((df[cols_pose] >= -1) & (df[cols_pose] <= 2)).mean(axis=1) > 0.80
    df = df[mask]

    # 4) Interpolación segura
    df[["ecg"] + cols_pose] = df[["ecg"] + cols_pose].interpolate(method="linear")

    # 5) Normalizar respecto a pelvis
    pelvis_x = df["23_x"]
    pelvis_y = df["23_y"]

    for i in range(33):
        df[f"{i}_x"] = df[f"{i}_x"] - pelvis_x
        df[f"{i}_y"] = df[f"{i}_y"] - pelvis_y

    # 6) Suavizado
    df["ecg_smooth"] = df["ecg"].rolling(5, center=True).median()
    for c in cols_pose:
        df[c] = df[c].rolling(3, center=True).mean()

    df = df.fillna(method="bfill").fillna(method="ffill").copy()

    # 7) Eliminar solo tramos planos
    planos = detectar_ecg_plano(df)
    df = df[~planos].copy()

    # =============================
    # RESTAURAR LABEL Y TIME
    # =============================
    if label_backup is not None:
        df["label"] = label_backup.loc[df.index]

    if time_backup is not None:
        df["time_sec"] = time_backup.loc[df.index]

    # =============================
    # ORDENAR COLUMNAS
    # =============================
    col_order = ["timestamp", "time_sec", "ecg", "ecg_smooth"] \
                + cols_pose + ["label"]

    df = df[col_order]

    # =============================
    # GUARDAR
    # =============================
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{os.path.splitext(os.path.basename(file))[0]}_limpio.csv"
    )
    df.to_csv(output_path, index=False)

    print(f"  ✔ Guardado en: {output_path}")
    print(f"  ✔ Filas finales: {len(df)}\n")
