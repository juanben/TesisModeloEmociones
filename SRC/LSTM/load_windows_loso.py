# ======================================================
# SCRIPT 1: Carga de ventanas + LOSO + Split 70/15/15
# Archivo: load_windows_loso.py
# ======================================================

import os
import glob
import numpy as np

WINDOWS_DIR = "../../Ventanas"
RANDOM_SEED = 42

def load_windows_by_subject(windows_dir):
    subjects_data = {}
    window_files = glob.glob(os.path.join(windows_dir, "*_windows.npy"))
    print("Buscando archivos en:", windows_dir)
    for wfile in window_files:
        subject = os.path.basename(wfile).split("_")[0]
        lfile = os.path.join(windows_dir, f"{subject}_labels.npy")

        if not os.path.exists(lfile):
            continue

        X = np.load(wfile)
        y = np.load(lfile)

        if len(X) != len(y):
            continue
        
        LABEL_MAP = {
            "neutro": 0,
            "miedo": 1,
            "ira": 2,
            "alegria": 3
        }

        y = np.array([LABEL_MAP[l] for l in y])    
        subjects_data[subject] = (X, y)

    return subjects_data


# ======================================================
# FUNCIÓN: LOSO con NORMALIZACIÓN GLOBAL POR FOLD
# ======================================================
def loso_split(subjects_data, subject_out,
               train_ratio=0.70, val_ratio=0.15, seed=42):

    # -----------------------------
    # 1) Extraer datos del sujeto de prueba (LOSO externo)
    # -----------------------------
    X_test_ext, y_test_ext = subjects_data[subject_out]

    # -----------------------------
    # 2) Concatenar el resto de sujetos
    # -----------------------------
    X_list = []
    y_list = []

    for sid, (X, y) in subjects_data.items():
        if sid == subject_out:
            continue
        X_list.append(X)
        y_list.append(y)

    X_rest = np.concatenate(X_list, axis=0)
    y_rest = np.concatenate(y_list, axis=0)

    # -----------------------------
    # 3) Mezclar train/val/test interno
    # -----------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_rest))
    X_rest = X_rest[idx]
    y_rest = y_rest[idx]

    n = len(y_rest)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_train = X_rest[:n_train]
    y_train = y_rest[:n_train]

    X_val = X_rest[n_train:n_train + n_val]
    y_val = y_rest[n_train:n_train + n_val]

    X_test_int = X_rest[n_train + n_val:]
    y_test_int = y_rest[n_train + n_val:]

    # =====================================================
    # 4) BALANCEO DEL TRAIN (oversampling + downsampling)
    # =====================================================

    rng = np.random.default_rng(seed)

    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))

    print("\nCounts antes de balanceo:", class_counts)

    # Target = promedio de clases
    target = int(np.mean(list(class_counts.values())))

    indices_final = []

    for cls in unique:
        idx_cls = np.where(y_train == cls)[0]
        count_cls = len(idx_cls)

        # ---- Oversampling ----
        if count_cls < target:
            num_extra = target - count_cls
            extra_indices = rng.choice(idx_cls, size=num_extra, replace=True)
            final_cls_idx = np.concatenate([idx_cls, extra_indices])

        # ---- Downsampling ----
        elif count_cls > int(1.5 * target):
            num_keep = int(1.5 * target)
            final_cls_idx = rng.choice(idx_cls, size=num_keep, replace=False)

        else:
            final_cls_idx = idx_cls

        indices_final.append(final_cls_idx)

    # Unir todas las clases
    indices_final = np.concatenate(indices_final)

    # Mezclar
    rng.shuffle(indices_final)

    # Aplicar a train
    X_train = X_train[indices_final]
    y_train = y_train[indices_final]

    # Mostrar counts balanceados
    u2, c2 = np.unique(y_train, return_counts=True)
    print("Counts DESPUÉS de balanceo:", dict(zip(u2, c2)))
    # =====================================================
    # 5) NORMALIZACIÓN GLOBAL DEL ECG SEGÚN TRAIN
    # =====================================================

    # ECG está en el canal 0
    ecg_train = X_train[:, :, 0]       # (N_train, seq_len)

    mean = ecg_train.mean()
    std  = ecg_train.std()

    if std == 0 or np.isnan(std):
        std = 1.0

    # ---- Aplicar normalización a TODOS los sets ----
    X_train[:, :, 0] = (X_train[:, :, 0] - mean) / std
    X_val[:, :, 0]   = (X_val[:, :, 0]   - mean) / std
    X_test_int[:, :, 0] = (X_test_int[:, :, 0] - mean) / std
    X_test_ext[:, :, 0] = (X_test_ext[:, :, 0] - mean) / std

    return X_train, y_train, X_val, y_val, X_test_int, y_test_int, X_test_ext, y_test_ext, mean, std
