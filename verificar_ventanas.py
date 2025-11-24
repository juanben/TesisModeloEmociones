import numpy as np
import glob
import os

WINDOW_DIR = "Ventanas"

for f in sorted(glob.glob(os.path.join(WINDOW_DIR, "*_labels.npy"))):
    subject = os.path.basename(f).replace("_labels.npy","")
    y = np.load(f)
    unique, counts = np.unique(y, return_counts=True)
    print(subject, dict(zip(unique, counts)))
