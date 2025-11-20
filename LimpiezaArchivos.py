import pandas as pd
import numpy as np
import glob
import os

# Carpeta destino
OUTPUT_DIR = "DatasetLimpio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Obtener solo archivos CSV que NO sean limpios
files = [f for f in glob.glob("DatasetSucio/*.csv") if "limpio" not in f.lower()]


# -------------------------------------------------------------------
#  Detectar tramos planos de ECG sin romper dimensiones
# -------------------------------------------------------------------
def detectar_ecg_plano(df, col="ecg_smooth", ventana=80, tol=2):

    valores = df[col].values
    n = len(valores)

    if n == 0 or n < ventana:
        return np.zeros(n, dtype=bool)

    dif = np.abs(np.diff(valores))          # len = n-1
    plano = (dif < tol).astype(int)

    conv = np.convolve(plano, np.ones(ventana), mode="same")  # len = n-1
    conv = conv >= ventana

    # Ajustar longitud al tamaño del DataFrame
    conv_full = np.zeros(n, dtype=bool)
    conv_full[:len(conv)] = conv

    return conv_full


# -------------------------------------------------------------------
#  Procesamiento archivo por archivo
# -------------------------------------------------------------------
for file in files:
    print(f"Procesando: {file}")
    df = pd.read_csv(file)

    # -----------------------------
    # 1. Eliminar frames sin pose
    # -----------------------------
    cols_pose = [c for c in df.columns if "_" in c and c != "ecg"]
    df = df[df[cols_pose].sum(axis=1) != 0]

    # -----------------------------
    # 2. Filtrar ECG basura
    # -----------------------------
    df = df[(df["ecg"] > 300) & (df["ecg"] < 3000)]

    # -----------------------------
    # 3. Filtrar poses fuera de rango
    # -----------------------------
    for c in cols_pose:
        df = df[df[c].between(-1, 2)]

    # -----------------------------
    # 4. Interpolación
    # -----------------------------
    df = df.interpolate(method="linear")

    # -----------------------------
    # 5. Normalizar respecto a pelvis (23)
    # -----------------------------
    pelvis_x = df["23_x"]
    pelvis_y = df["23_y"]

    for i in range(33):
        df[f"{i}_x"] = df[f"{i}_x"] - pelvis_x
        df[f"{i}_y"] = df[f"{i}_y"] - pelvis_y

    # -----------------------------
    # 6. Suavizado
    # -----------------------------
    df["ecg_smooth"] = df["ecg"].rolling(5, center=True).median()

    for c in cols_pose:
        df[c] = df[c].rolling(3, center=True).mean()

    df = df.dropna().copy()    # <--- FIX: elimina fragmentación

    # -----------------------------
    # 7. Detectar tramos planos
    # -----------------------------
    planos = detectar_ecg_plano(df)
    df = df[~planos].copy()

    # -----------------------------
    # 8. Guardar dataset limpio
    # -----------------------------
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{os.path.splitext(os.path.basename(file))[0]}_limpio.csv"
    )
    df.to_csv(output_path, index=False)

    print(f"  ✔ Guardado en: {output_path}\n")
