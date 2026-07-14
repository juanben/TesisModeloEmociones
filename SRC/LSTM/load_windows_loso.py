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
    """
    Carga de forma dinámica y absoluta todos los archivos .npy 
    desde el directorio especificado.
    """
    import os
    import numpy as np
    
    # Forzar a que use la ruta absoluta que le envía run_loso.py
    windows_dir = os.path.abspath(windows_dir)
    print(f"🔍 Cargando archivos .npy reales desde: {windows_dir}")
    
    subjects_data = {}
    
    if not os.path.exists(windows_dir):
        print(f"❌ ERROR: La carpeta {windows_dir} no existe.")
        return subjects_data

    # Listar los archivos usando directamente la ruta absoluta corregida
    files = os.listdir(windows_dir)
    
    for f in files:
        if f.endswith("_windows.npy"):
            subject_id = f.split("_")[0]
            
            win_path = os.path.join(windows_dir, f)
            lbl_path = os.path.join(windows_dir, f.replace("_windows.npy", "_labels.npy"))
            
            if os.path.exists(lbl_path):
                X = np.load(win_path, allow_pickle=True)
                y = np.load(lbl_path, allow_pickle=True)
                
                # Omitir si la matriz de ventanas quedó vacía tras los filtros
                if len(X.shape) != 3 or X.shape[0] == 0:
                    continue
                    
                # Corregir mapeos de tipo de datos numéricos enteros
                y = np.array(y, dtype=np.int64)
                
                subjects_data[subject_id] = (X, y)
                
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
