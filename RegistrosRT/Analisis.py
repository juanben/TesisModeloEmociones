"""
analyze_emotions_csvs.py

Lee múltiples CSVs con columnas:
timestamp, label, conf, p_neutro, p_miedo, p_ira, p_alegria

Genera:
- summary_by_file.csv
- label_counts_by_file.csv
- summary_overall.csv
- Plots PNG en ./outputs/plots/

Uso:
  python analyze_emotions_csvs.py --input_dir ./mis_csvs
  python analyze_emotions_csvs.py --files byron.csv ana.csv pedro.csv ...

Requisitos:
  pip install pandas matplotlib
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt


EXPECTED_COLS = {"timestamp", "label", "conf"}
PROB_PREFIX = "p_"


def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Archivo {path.name}: faltan columnas requeridas: {sorted(missing)}")

    # Convertir numéricos
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce")

    # Quitar filas inválidas
    df = df.dropna(subset=["timestamp", "label", "conf"]).copy()

    # Normalizar label (por si viene con espacios)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Detectar columnas de probabilidad p_*
    prob_cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    for c in prob_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convertir timestamp (epoch seconds) a datetime
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce", utc=True)
    # si quieres hora local Ecuador, descomenta:
    # df["dt"] = df["dt"].dt.tz_convert("America/Guayaquil")

    return df


def add_file_metadata(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    out = df.copy()
    out["file"] = filename
    out["subject"] = Path(filename).stem  # por defecto: nombre del archivo sin extensión
    return out


def summarize_file(df: pd.DataFrame) -> dict:
    # Distribución de labels
    counts = df["label"].value_counts(dropna=False).to_dict()
    total = int(df.shape[0])

    # Estadísticos de confianza
    conf_mean = float(df["conf"].mean())
    conf_std = float(df["conf"].std(ddof=1)) if total > 1 else 0.0
    conf_med = float(df["conf"].median())

    # Duración (si hay datetimes válidos)
    if df["dt"].notna().any():
        t0 = df["dt"].min()
        t1 = df["dt"].max()
        duration_s = float((t1 - t0).total_seconds())
    else:
        duration_s = float("nan")

    # Transiciones de emoción (cambios de label en el tiempo)
    df_sorted = df.sort_values("timestamp")
    transitions = int((df_sorted["label"] != df_sorted["label"].shift(1)).sum() - 1) if total > 1 else 0

    # Entropía promedio de probabilidades (si existen p_*)
    prob_cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    entropy_mean = float("nan")
    if prob_cols:
        p = df[prob_cols].copy()
        # Evitar log(0)
        p = p.clip(lower=1e-12)
        # Normalizar por fila por si no suma 1 perfecto
        p = p.div(p.sum(axis=1), axis=0)
        entropy = -(p * p.applymap(lambda x: float(pd.np.log(x)))).sum(axis=1)  # nats
        entropy_mean = float(entropy.mean())

    return {
        "n_samples": total,
        "duration_s": duration_s,
        "conf_mean": conf_mean,
        "conf_median": conf_med,
        "conf_std": conf_std,
        "label_transitions": transitions,
        "entropy_mean_nats": entropy_mean,
        **{f"count_{k}": int(v) for k, v in counts.items()},
    }


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_label_distribution(all_df: pd.DataFrame, outdir: Path) -> None:
    counts = all_df["label"].value_counts()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Distribución de emociones (global)")
    plt.xlabel("label")
    plt.ylabel("conteo")
    plt.tight_layout()
    plt.savefig(outdir / "label_distribution_global.png", dpi=200)
    plt.close()


def plot_conf_by_file(all_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    all_df.boxplot(column="conf", by="file", rot=45)
    plt.title("Confianza por archivo")
    plt.suptitle("")  # quita título automático de pandas
    plt.xlabel("archivo")
    plt.ylabel("conf")
    plt.tight_layout()
    plt.savefig(outdir / "conf_by_file.png", dpi=200)
    plt.close()


def plot_conf_timeseries(df: pd.DataFrame, outdir: Path, file_name: str) -> None:
    df_sorted = df.sort_values("timestamp")
    plt.figure()
    plt.plot(df_sorted["timestamp"], df_sorted["conf"])
    plt.title(f"Confianza vs tiempo - {file_name}")
    plt.xlabel("timestamp (epoch s)")
    plt.ylabel("conf")
    plt.tight_layout()
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", Path(file_name).stem)
    plt.savefig(outdir / f"conf_timeseries_{safe_name}.png", dpi=200)
    plt.close()


def plot_prob_area(df: pd.DataFrame, outdir: Path, file_name: str) -> None:
    prob_cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    if not prob_cols:
        return
    df_sorted = df.sort_values("timestamp")
    plt.figure()
    plt.stackplot(df_sorted["timestamp"], *[df_sorted[c].fillna(0.0) for c in prob_cols], labels=prob_cols)
    plt.title(f"Probabilidades por clase (area) - {file_name}")
    plt.xlabel("timestamp (epoch s)")
    plt.ylabel("prob")
    plt.legend(loc="upper right")
    plt.tight_layout()
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", Path(file_name).stem)
    plt.savefig(outdir / f"prob_area_{safe_name}.png", dpi=200)
    plt.close()


def main(files: Optional[List[str]], input_dir: Optional[str], out_dir: str) -> None:
    out_base = Path(out_dir)
    plots_dir = out_base / "plots"
    ensure_dir(out_base)
    ensure_dir(plots_dir)

    paths: List[Path] = []
    if files:
        paths = [Path(f) for f in files]
    elif input_dir:
        d = Path(input_dir)
        paths = sorted(d.glob("*.csv"))
    else:
        raise ValueError("Debes pasar --files o --input_dir")

    if not paths:
        raise ValueError("No encontré CSVs para analizar.")

    all_dfs = []
    per_file_rows = []

    for p in paths:
        df = safe_read_csv(p)
        df = add_file_metadata(df, p.name)
        all_dfs.append(df)

        stats = summarize_file(df)
        stats["file"] = p.name
        per_file_rows.append(stats)

        # plots por archivo
        plot_conf_timeseries(df, plots_dir, p.name)
        plot_prob_area(df, plots_dir, p.name)

    all_df = pd.concat(all_dfs, ignore_index=True)

    # --- Tablas ---
    summary_by_file = pd.DataFrame(per_file_rows).sort_values("file")
    summary_by_file.to_csv(out_base / "summary_by_file.csv", index=False, encoding="utf-8-sig")

    label_counts_by_file = (
        all_df.pivot_table(index="file", columns="label", values="conf", aggfunc="size", fill_value=0)
        .reset_index()
    )
    label_counts_by_file.to_csv(out_base / "label_counts_by_file.csv", index=False, encoding="utf-8-sig")

    # Resumen global
    global_stats = summarize_file(all_df)
    summary_overall = pd.DataFrame([{"metric": k, "value": v} for k, v in global_stats.items()])
    summary_overall.to_csv(out_base / "summary_overall.csv", index=False, encoding="utf-8-sig")

    # --- Plots globales ---
    plot_label_distribution(all_df, plots_dir)
    plot_conf_by_file(all_df, plots_dir)

    print("✅ Listo. Resultados en:", str(out_base.resolve()))
    print(" - summary_by_file.csv")
    print(" - label_counts_by_file.csv")
    print(" - summary_overall.csv")
    print(" - plots/*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Carpeta con CSVs (ej: ./mis_csvs)")
    parser.add_argument("--files", nargs="+", default=None, help="Lista de archivos CSV")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Carpeta de salida")
    args = parser.parse_args()

    main(files=args.files, input_dir=args.input_dir, out_dir=args.out_dir)