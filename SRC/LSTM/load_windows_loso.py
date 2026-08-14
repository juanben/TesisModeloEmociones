# ======================================================
# SCRIPT 1: Carga de ventanas + LOSO + Split agrupado por sujeto
#           + Normalización Z-score POR SUJETO
# Archivo: load_windows_loso.py
#
# CAMBIOS RESPECTO A LA VERSIÓN ORIGINAL:
#   1) El split interno 70/15/15 ya NO mezcla ventanas de los 19
#      sujetos al azar. Ahora se seleccionan SUJETOS COMPLETOS
#      para train / val / test_int, evitando que ventanas
#      solapadas al 75% del mismo sujeto queden repartidas entre
#      conjuntos distintos (nested validation leakage).
#   2) La normalización Z-score del canal ECG (canal 0) ya NO se
#      calcula de forma global sobre el pool de entrenamiento.
#      Ahora se calcula de forma INDEPENDIENTE para cada sujeto,
#      usando únicamente los datos de ese sujeto (coherente con
#      la Ecuación 1 del paper). El sujeto de test externo (LOSO)
#      se normaliza con SU PROPIO mean/std, nunca con estadísticos
#      ajenos.
# ======================================================

import os
import numpy as np

WINDOWS_DIR = "../../Ventanas"
RANDOM_SEED = 42


def load_windows_by_subject(windows_dir):
    """
    Carga de forma dinámica y absoluta todos los archivos .npy
    desde el directorio especificado.
    (Sin cambios respecto a la versión original)
    """
    windows_dir = os.path.abspath(windows_dir)
    print(f"🔍 Cargando archivos .npy reales desde: {windows_dir}")

    subjects_data = {}

    if not os.path.exists(windows_dir):
        print(f"❌ ERROR: La carpeta {windows_dir} no existe.")
        return subjects_data

    files = os.listdir(windows_dir)

    for f in files:
        if f.endswith("_windows.npy"):
            subject_id = f.split("_")[0]

            win_path = os.path.join(windows_dir, f)
            lbl_path = os.path.join(windows_dir, f.replace("_windows.npy", "_labels.npy"))

            if os.path.exists(lbl_path):
                X = np.load(win_path, allow_pickle=True)
                y = np.load(lbl_path, allow_pickle=True)

                if len(X.shape) != 3 or X.shape[0] == 0:
                    continue

                y = np.array(y, dtype=np.int64)

                subjects_data[subject_id] = (X, y)

    return subjects_data


# ======================================================
# NORMALIZACIÓN Z-SCORE POR SUJETO (canal ECG = canal 0)
# ======================================================
def compute_subject_ecg_stats(X_subject):
    """
    Calcula mean y std del canal ECG (canal 0) usando
    EXCLUSIVAMENTE los datos de un sujeto individual.
    """
    ecg = X_subject[:, :, 0]
    mean = ecg.mean()
    std = ecg.std()
    if std == 0 or np.isnan(std):
        std = 1.0
    return mean, std


def normalize_subject_ecg(X_subject, mean, std):
    """
    Aplica normalización Z-score al canal ECG (canal 0)
    de un sujeto, usando el mean/std provisto.
    No modifica el array original (devuelve copia).
    """
    X_out = X_subject.copy()
    X_out[:, :, 0] = (X_out[:, :, 0] - mean) / std
    return X_out


# ======================================================
# FUNCIÓN: LOSO con SPLIT AGRUPADO POR SUJETO
#          + NORMALIZACIÓN Z-SCORE POR SUJETO
# ======================================================
def loso_split(subjects_data, subject_out,
                train_ratio=0.70, val_ratio=0.15, seed=42):

    # -----------------------------
    # 1) Extraer datos del sujeto de prueba (LOSO externo)
    # -----------------------------
    X_test_ext_raw, y_test_ext = subjects_data[subject_out]

    # Normalización POR SUJETO del test externo: usa
    # exclusivamente sus propios datos (nunca estadísticos
    # de otros sujetos).
    mean_ext, std_ext = compute_subject_ecg_stats(X_test_ext_raw)
    X_test_ext = normalize_subject_ecg(X_test_ext_raw, mean_ext, std_ext)

    # -----------------------------
    # 2) Determinar los 19 sujetos restantes
    # -----------------------------
    remaining_subjects = [sid for sid in subjects_data.keys() if sid != subject_out]

    rng = np.random.default_rng(seed)
    shuffled_subjects = rng.permutation(remaining_subjects)

    n_subj = len(shuffled_subjects)
    n_val_subj = max(1, round(val_ratio * n_subj))
    n_test_subj = max(1, round((1 - train_ratio - val_ratio) * n_subj))
    n_train_subj = n_subj - n_val_subj - n_test_subj

    train_subjects = list(shuffled_subjects[:n_train_subj])
    val_subjects = list(shuffled_subjects[n_train_subj:n_train_subj + n_val_subj])
    test_int_subjects = list(shuffled_subjects[n_train_subj + n_val_subj:])

    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects   ({len(val_subjects)}): {val_subjects}")
    print(f"Test_int subjects ({len(test_int_subjects)}): {test_int_subjects}")

    # -----------------------------
    # 3) Normalizar CADA sujeto con SUS PROPIOS estadísticos
    #    y luego agrupar por conjunto (train / val / test_int)
    # -----------------------------
    def gather_normalized(subj_list):
        Xs, ys = [], []
        for sid in subj_list:
            X_raw, y_raw = subjects_data[sid]
            m, s = compute_subject_ecg_stats(X_raw)
            X_norm = normalize_subject_ecg(X_raw, m, s)
            Xs.append(X_norm)
            ys.append(y_raw)
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = gather_normalized(train_subjects)
    X_val, y_val = gather_normalized(val_subjects)
    X_test_int, y_test_int = gather_normalized(test_int_subjects)

    # -----------------------------
    # 4) Mezclar ventanas DENTRO de train (esto ya no causa fuga,
    #    porque no hay overlap entre sujetos de train y de val/test)
    # -----------------------------
    idx_train = rng.permutation(len(y_train))
    X_train = X_train[idx_train]
    y_train = y_train[idx_train]

    # =====================================================
    # 5) BALANCEO DEL TRAIN (oversampling + downsampling)
    #    Se calcula EXCLUSIVAMENTE sobre y_train (partición
    #    interna de entrenamiento), sin tocar val ni test_int.
    # =====================================================
    rng_bal = np.random.default_rng(seed)

    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("\nCounts antes de balanceo:", class_counts)

    target = int(np.mean(list(class_counts.values())))

    indices_final = []
    for cls in unique:
        idx_cls = np.where(y_train == cls)[0]
        count_cls = len(idx_cls)

        if count_cls < target:
            num_extra = target - count_cls
            extra_indices = rng_bal.choice(idx_cls, size=num_extra, replace=True)
            final_cls_idx = np.concatenate([idx_cls, extra_indices])
        elif count_cls > int(1.5 * target):
            num_keep = int(1.5 * target)
            final_cls_idx = rng_bal.choice(idx_cls, size=num_keep, replace=False)
        else:
            final_cls_idx = idx_cls

        indices_final.append(final_cls_idx)

    indices_final = np.concatenate(indices_final)
    rng_bal.shuffle(indices_final)

    X_train = X_train[indices_final]
    y_train = y_train[indices_final]

    u2, c2 = np.unique(y_train, return_counts=True)
    print("Counts DESPUÉS de balanceo:", dict(zip(u2, c2)))

    # NOTA: ya NO se recalcula mean/std global aquí -- cada
    # sujeto fue normalizado individualmente en el paso 3,
    # antes del balanceo y antes de la mezcla.

    # Se retorna mean_ext, std_ext solo para fines de logging /
    # trazabilidad del sujeto de test externo (no se usa para
    # normalizar ningún otro conjunto).
    return (X_train, y_train,
            X_val, y_val,
            X_test_int, y_test_int,
            X_test_ext, y_test_ext,
            mean_ext, std_ext)