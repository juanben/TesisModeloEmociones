import os, glob
import numpy as np

WINDOW_DIR = "Ventanas"

def inspect_subject(subject):
    x_path = os.path.join(WINDOW_DIR, f"{subject}_windows.npy")
    y_path = os.path.join(WINDOW_DIR, f"{subject}_labels.npy")

    X = np.load(x_path)   # (N, T, F)
    y = np.load(y_path)

    N, T, F = X.shape
    unique, counts = np.unique(y, return_counts=True)

    print(f"\n=== {subject} ===")
    print(f"X shape: {X.shape}  -> N={N}, T={T}, F={F}")
    print("labels:", dict(zip(unique, counts)))

    # stats por feature (para detectar 3 features “raras”)
    feat_mean = X.reshape(-1, F).mean(axis=0)
    feat_std  = X.reshape(-1, F).std(axis=0)
    print("Primeras 10 std:", np.round(feat_std[:10], 4))

    # detectar features casi constantes (std muy bajo)
    low_var = np.where(feat_std < 1e-6)[0]
    if len(low_var) > 0:
        print("⚠️ Features casi constantes:", low_var[:20], "..." if len(low_var) > 20 else "")

    return F

# -------------------------
# recorrer todos los sujetos
# -------------------------
all_x = sorted(glob.glob(os.path.join(WINDOW_DIR, "*_windows.npy")))
features_set = set()

for xp in all_x:
    subject = os.path.basename(xp).replace("_windows.npy", "")
    F = inspect_subject(subject)
    features_set.add(F)

print("\n==============================")
print("Features detectadas en todos los sujetos:", sorted(features_set))
print("==============================\n")
