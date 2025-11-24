import pandas as pd
import os
import re

# === CONFIG ===
PLANTILLA = "plantilla_tiempos.xlsx"
DATA_FOLDER = "DatasetSucio"
OUTPUT_FOLDER = "datasets_etiquetados"

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def time_to_seconds(t):
    t = str(t).strip().lower()
    if t == "final":
        return None
    parts = re.split("[:.]", t)
    parts = [int(p) for p in parts]
    if len(parts) == 2:
        m, s = parts
        return m*60 + s
    elif len(parts) == 3:
        h, m, s = parts
        return h*3600 + m*60 + s
    return None

plantilla = pd.read_excel(PLANTILLA)
plantilla["emocion_principal"] = plantilla["emocion_principal"].str.lower().str.strip()

for archivo in plantilla["archivo_csv"].unique():

    file_path = os.path.join(DATA_FOLDER, archivo)

    if not os.path.exists(file_path):
        print(f"⚠ Archivo no encontrado: {archivo}")
        continue

    print(f"✅ Procesando: {archivo}")

    df = pd.read_csv(file_path)

    # Convertir timestamps absolutos a tiempo relativo
    df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]

    intervals = plantilla[plantilla["archivo_csv"] == archivo]

    df["label"] = "neutro"

    for _, row in intervals.iterrows():
        start = time_to_seconds(row["inicio_emocion"])
        end = time_to_seconds(row["fin_registro"])
        emotion = row["emocion_principal"]

        if end is None:
            end = df["time_sec"].max()

        mask = (df["time_sec"] >= start) & (df["time_sec"] <= end)
        df.loc[mask, "label"] = emotion

    out = os.path.join(OUTPUT_FOLDER, archivo.replace(".csv", "_etiquetado.csv"))
    df.to_csv(out, index=False)

print("\n🎉 ETIQUETADO COMPLETADO")