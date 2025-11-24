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


def loso_split(subjects_data, subject_out,
               train_ratio=0.70, val_ratio=0.15, seed=42):

    X_test_ext, y_test_ext = subjects_data[subject_out]

    X_list = []
    y_list = []

    for sid, (X, y) in subjects_data.items():
        if sid == subject_out:
            continue
        X_list.append(X)
        y_list.append(y)

    X_rest = np.concatenate(X_list, axis=0)
    y_rest = np.concatenate(y_list, axis=0)

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

    return X_train, y_train, X_val, y_val, X_test_int, y_test_int, X_test_ext, y_test_ext
