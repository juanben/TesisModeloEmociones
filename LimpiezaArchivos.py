# ======================================================================
# ARCHIVO: LimpiezaArchivos_2.py (Versión Calibrada y Segura)
# ======================================================================
import pandas as pd
import numpy as np
import glob
import os
import re

OUTPUT_DIR = "DatasetLimpio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in glob.glob("datasets_etiquetados/*.csv")]

def detectar_ecg_plano(df, col="ecg_smooth", ventana=25, tol=3):
    valores = df[col].values
    n = len(valores)
    if n == 0 or n < ventana:
        return np.zeros(n, dtype=bool)
    dif = np.abs(np.diff(valores))
    plano = (dif < tol).astype(int)
    conv = np.convolve(plano, np.ones(ventana), mode="same")
    conv_bool = conv >= ventana
    conv_full = np.zeros(n, dtype=bool)
    conv_full[:len(conv_bool)] = conv_bool
    return conv_full

def normalize_pose(df):
    shoulder_x = (df["11_x"] + df["12_x"]) / 2
    shoulder_y = (df["11_y"] + df["12_y"]) / 2
    pelvis_x = (df["23_x"] + df["24_x"]) / 2
    pelvis_y = (df["23_y"] + df["24_y"]) / 2
    vx = shoulder_x - pelvis_x
    vy = shoulder_y - pelvis_y
    angles = np.arctan2(vy, vx)
    cos_a = np.cos(-angles)
    sin_a = np.sin(-angles)

    for i in range(33):
        x = df[f"{i}_x"].values - pelvis_x
        y = df[f"{i}_y"].values - pelvis_y
        df[f"{i}_x"] = x * cos_a - y * sin_a
        df[f"{i}_y"] = x * sin_a + y * cos_a

    hip_dist = np.sqrt((df["23_x"] - df["24_x"])**2 + (df["23_y"] - df["24_y"])**2)
    hip_dist[hip_dist == 0] = 1e-6
    for i in range(33):
        df[f"{i}_x"] /= hip_dist
        df[f"{i}_y"] /= hip_dist
    return df

for file in files:
    print(f"Procesando de forma calibrada: {file}")
    df = pd.read_csv(file).reset_index(drop=True)
    cols_pose = [c for c in df.columns if re.match(r"\d+_[xyz]$", c)]

    df = df[(df[cols_pose].abs().sum(axis=1) != 0)]
    df = df.dropna(subset=cols_pose)
    df = df[(df["ecg"] > 100) & (df["ecg"] < 4095)]

    mask = ((df[cols_pose] >= -1) & (df[cols_pose] <= 2)).mean(axis=1) > 0.80
    df = df[mask].copy()
    df[["ecg"] + cols_pose] = df[["ecg"] + cols_pose].interpolate(method="linear")
    df = normalize_pose(df)

    df["ecg_smooth"] = df["ecg"].rolling(5, center=True).median()
    df = df.copy()
    
    for c in cols_pose:
        df[c] = df[c].rolling(3, center=True).mean()

    df = df.bfill().ffill().copy()
    planos = detectar_ecg_plano(df)
    df = df[~planos].copy()

    # Calibración del filtro dinámico para evitar el vaciado de clases
    ecg_diff = df["ecg_smooth"].diff().fillna(0)
    ecg_accel = ecg_diff.diff().abs().fillna(0)
    
    # Filtro suave (Percentil 25): solo remueve ruido de parálisis del sensor o latencias basales reales
    umbral_ruido = ecg_accel.quantile(0.25)

    labels_originales = df["label"].copy().values
    labels_filtrados = []

    for idx, label_act in enumerate(labels_originales):
        if label_act in ["miedo", "ira", "alegria"]:
            if ecg_accel.iloc[idx] <= umbral_ruido:
                labels_filtrados.append("neutro")  # Reclasifica solo si la señal está muerta
            else:
                labels_filtrados.append(label_act) # Mantiene la etiqueta emocional legítima
        else:
            labels_filtrados.append("neutro")

    df["label"] = labels_filtrados
    col_order = ["timestamp", "time_sec", "ecg", "ecg_smooth"] + cols_pose + ["label"]
    df = df[col_order]

    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(file))[0]}_limpio.csv")
    df.to_csv(output_path, index=False)