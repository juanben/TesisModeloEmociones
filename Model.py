# multimodal_pipeline_and_model.py
# Requisitos: numpy, pandas, scipy, sklearn, torch
import os
import math
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Tuple

# -------------------------
# CONFIGURACIÓN (ajusta)
# -------------------------
FS = 250                # frecuencia objetivo ECG (Hz)
WINDOW_SEC = 5          # segundos por ventana
OVERLAP = 0.5           # solapamiento (0..1)
N_CLASSES = 3           # número de emociones (ej: alegria, ira, miedo)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# mapea etiquetas string a int
LABEL_MAP = {'alegria': 0, 'ira': 1, 'miedo': 2}

# timeline de escenas (segundos). Ajustar según tu juego.
SCENES = [
    (0, 60, 'alegria'),
    (60, 120, 'ira'),
    (120, 180, 'miedo')
]

# -------------------------
# UTILIDADES DE SEÑAL
# -------------------------
def parse_timestamp_to_seconds(ts_raw):
    """
    Limpia timestamps como '17.592.026.038.691.400' -> nanoseconds int -> seconds float.
    Ajusta según tu formato real.
    """
    s = str(ts_raw).replace('.', '')
    try:
        val = int(s)
        return val / 1e9  # si está en ns
    except:
        try:
            return float(ts_raw)
        except:
            raise ValueError("Timestamp no convertible: " + str(ts_raw))

def bandpass_filter(sig, fs, low=0.5, high=40, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

# -------------------------
# DATASET dinámico (ventanas)
# -------------------------
class VRMultimodalDataset(Dataset):
    """
    Lee CSV (fila por fila), sincroniza ECG y pose (si están en mismo CSV),
    hace resample del ECG a FS, filtra (opcional) y ventana dinámicamente.
    CSV expected columns: timestamp, ecg, [pose keypoints...], label (opcional), participant_id (opcional)
    """
    def __init__(self, csv_path: str,
                 fs: int = FS,
                 window_sec: float = WINDOW_SEC,
                 overlap: float = OVERLAP,
                 scenes: List[Tuple[float,float,str]] = SCENES,
                 label_map: dict = LABEL_MAP,
                 do_filter: bool = True,
                 pose_columns: List[str] = None):
        df = pd.read_csv(csv_path, sep=None, engine='python')  # autodetect delimiter
        # Normalizar nombres
        df.columns = [c.strip() for c in df.columns]
        # parse timestamps
        df['timestamp_s'] = df['timestamp'].apply(parse_timestamp_to_seconds)
        df = df.sort_values('timestamp_s').reset_index(drop=True)
        # crear label por fila según scenes (si no existe columna label)
        if 'label' not in df.columns:
            df['label'] = df['timestamp_s'].apply(lambda t: self._label_from_scenes(t, scenes))
        # opcionales
        if 'participant_id' not in df.columns:
            df['participant_id'] = 0  # si no hay múltiples participantes, 0 por defecto

        # ECG vector y resampleo por sesión (asumimos que el csv ya tiene muestras de ECG)
        ecg_raw = df['ecg'].values.astype(float)
        # timestamp relativo (segundos desde inicio)
        t0 = df['timestamp_s'].iloc[0]
        times = df['timestamp_s'].values - t0
        duration = times[-1] - times[0]
        # resample ECG a fs uniformly
        target_len = int(np.ceil(duration * fs)) + 1
        new_times = np.linspace(times[0], times[-1], target_len)
        ecg_resampled = np.interp(new_times, times, ecg_raw)
        if do_filter:
            try:
                ecg_resampled = bandpass_filter(ecg_resampled, fs)
            except Exception as e:
                print("Warning: filter failed:", e)

        # pose: if present, resample/interpolate pose columns to ecg timeline
        pose_cols = pose_columns or [c for c in df.columns if c.startswith('0_') or c.startswith('pose')]
        pose_resampled = None
        if len(pose_cols) > 0:
            pose_matrix = df[pose_cols].astype(float).values
            # map original times -> pose (may have fewer samples); do interpolation per column
            pose_resampled = np.zeros((len(new_times), pose_matrix.shape[1]))
            for i in range(pose_matrix.shape[1]):
                pose_resampled[:, i] = np.interp(new_times, times, pose_matrix[:, i])

        # create windowed arrays
        self.fs = fs
        self.window_size = int(window_sec * fs)
        self.step = int(self.window_size * (1 - overlap))
        self.ecg_windows = []
        self.pose_windows = []
        self.labels = []
        self.participants = []

        num = len(ecg_resampled)
        for start in range(0, num - self.window_size + 1, self.step):
            end = start + self.window_size
            seg_ecg = ecg_resampled[start:end]
            # timestamp center of window
            t_center = new_times[start + self.window_size//2] + t0
            label = self._label_from_scenes(t_center, scenes)
            if label is None:
                continue
            if label not in label_map:
                continue
            self.ecg_windows.append(seg_ecg.astype(np.float32))
            if pose_resampled is not None:
                # simple: mean pose vector in window; could keep sequence for LSTM
                self.pose_windows.append(pose_resampled[start:end].astype(np.float32))
            else:
                self.pose_windows.append(np.zeros((self.window_size, 0), dtype=np.float32))
            self.labels.append(label_map[label])
            self.participants.append(df['participant_id'].iloc[0])

        self.ecg_windows = np.stack(self.ecg_windows)  # (N, window)
        self.pose_windows = np.stack(self.pose_windows)  # (N, window, pose_dim)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.participants = np.array(self.participants)

        # scaler ECG per dataset (mejor normalizar por sujeto en production)
        self.ecg_mean = self.ecg_windows.mean()
        self.ecg_std = self.ecg_windows.std() + 1e-8
        self.ecg_windows = (self.ecg_windows - self.ecg_mean) / self.ecg_std

    def _label_from_scenes(self, t_sec, scenes):
        for a,b,l in scenes:
            if a <= t_sec < b:
                return l
        return None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # returns tensors: ecg (1, L), pose (L, P) or pooled (P,)
        ecg = torch.from_numpy(self.ecg_windows[idx]).unsqueeze(0)  # (1, L)
        pose_seq = torch.from_numpy(self.pose_windows[idx])  # (L, P)
        label = int(self.labels[idx])
        # opcional: reducir pose_seq a vector (mean pooling)
        if pose_seq.shape[1] == 0:
            pose_vec = torch.zeros(0)
        else:
            pose_vec = pose_seq.mean(dim=0)
        return ecg.float(), pose_vec.float(), label, int(self.participants[idx])

# -------------------------
# MODELO multimodal
# -------------------------
class ECGBranch(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)  # (batch, 128)

class PoseBranch(nn.Module):
    def __init__(self, pose_dim):
        super().__init__()
        if pose_dim <= 0:
            # dummy identity
            self.net = nn.Identity()
            self.out_dim = 0
        else:
            self.net = nn.Sequential(
                nn.Linear(pose_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.out_dim = 64

    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.numel() == 0:
            # empty pose vector
            return x.new_zeros((x.shape[0], 0))
        return self.net(x)

class FusionModel(nn.Module):
    def __init__(self, pose_dim, n_classes=N_CLASSES):
        super().__init__()
        self.ecg_branch = ECGBranch(in_channels=1)
        self.pose_branch = PoseBranch(pose_dim)
        fuse_dim = 128 + (self.pose_branch.out_dim if self.pose_branch.out_dim > 0 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, ecg, pose):
        # ecg: (B,1,L) ; pose: (B, P)
        e = self.ecg_branch(ecg)       # (B,128)
        p = self.pose_branch(pose)     # (B,Pout) or (B,0)
        if p.shape[1] == 0:
            x = e
        else:
            x = torch.cat([e, p], dim=1)
        return self.classifier(x)

# -------------------------
# TRAIN / EVAL helper
# -------------------------
def train_epoch(model, loader, opt, criterion):
    model.train()
    losses = []
    preds, trues = [], []
    for ecg, pose, labels, _ in loader:
        ecg = ecg.to(DEVICE)
        pose = pose.to(DEVICE)
        labels = labels.to(DEVICE)
        out = model(ecg, pose)
        loss = criterion(out, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds += out.argmax(dim=1).detach().cpu().numpy().tolist()
        trues += labels.detach().cpu().numpy().tolist()
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    return np.mean(losses), acc, f1

def eval_epoch(model, loader, criterion):
    model.eval()
    losses = []
    preds, trues = [], []
    with torch.no_grad():
        for ecg, pose, labels, _ in loader:
            ecg = ecg.to(DEVICE)
            pose = pose.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model(ecg, pose)
            loss = criterion(out, labels)
            losses.append(loss.item())
            preds += out.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
    acc = accuracy_score(trues, preds) if len(trues)>0 else 0.0
    f1 = f1_score(trues, preds, average='macro') if len(trues)>0 else 0.0
    return np.mean(losses), acc, f1, confusion_matrix(trues, preds) if len(trues)>0 else None

# -------------------------
# EJEMPLO de uso
# -------------------------
def example_train(csv_path, epochs=30, batch_size=32, lr=1e-3):
    ds = VRMultimodalDataset(csv_path, fs=FS, window_sec=WINDOW_SEC, overlap=OVERLAP,
                             scenes=SCENES, label_map=LABEL_MAP, do_filter=True)
    print("Windows:", len(ds))
    # split simple: stratified by label (si tienes pocos participantes). Mejor: LOSO por participant
    # aqui haremos split random
    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)

    pose_dim = ds.pose_windows.shape[2] if ds.pose_windows.ndim==3 else 0
    model = FusionModel(pose_dim=pose_dim, n_classes=len(LABEL_MAP)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_f1 = 0.0
    for ep in range(1, epochs+1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, opt, criterion)
        val_loss, val_acc, val_f1, cm = eval_epoch(model, val_loader, criterion)
        print(f"Epoch {ep}/{epochs} | tr_loss={train_loss:.4f} tr_acc={train_acc:.3f} tr_f1={train_f1:.3f} |"
              f" val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({'model_state': model.state_dict(), 'scaler_mean': ds.ecg_mean, 'scaler_std': ds.ecg_std},
                       'best_model.pth')
    print("Mejor F1 validación:", best_val_f1)

# -------------------------
# LO SO (Leave-One-Subject-Out) quick helper
# -------------------------
def loso_train(csv_paths: List[str], epochs=20, batch_size=32, lr=1e-3):
    """
    Si tienes varios CSV (uno por participante), puedes hacer LOSO: iterar each csv as test.
    csv_paths: list of csv files (one per participant)
    """
    results = []
    for i, test_csv in enumerate(csv_paths):
        print("LOSO iter:", i, "test:", test_csv)
        # combinar todos excepto test en train
        train_csvs = [p for j,p in enumerate(csv_paths) if j!=i]
        # build datasets
        train_ds_list = [VRMultimodalDataset(p, scenes=SCENES, label_map=LABEL_MAP) for p in train_csvs]
        train_ecg = np.concatenate([d.ecg_windows for d in train_ds_list], axis=0)
        train_pose = np.concatenate([d.pose_windows for d in train_ds_list], axis=0)
        train_labels = np.concatenate([d.labels for d in train_ds_list], axis=0)
        # make a lightweight combined dataset object
        class SmallDS(Dataset):
            def __init__(self, ecg, pose, labels):
                self.ecg = ecg
                self.pose = pose
                self.labels = labels
            def __len__(self): return len(self.labels)
            def __getitem__(self, idx):
                ecg = torch.from_numpy(self.ecg[idx]).unsqueeze(0)
                pose_vec = torch.from_numpy(self.pose[idx].mean(axis=0)) if self.pose.shape[1]>0 else torch.zeros(0)
                return ecg.float(), pose_vec.float(), int(self.labels[idx]), 0
        train_ds = SmallDS(train_ecg, train_pose, train_labels)
        test_ds = VRMultimodalDataset(test_csv, scenes=SCENES, label_map=LABEL_MAP)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        pose_dim = train_ds.pose.shape[2] if train_ds.pose.ndim==3 else 0
        model = FusionModel(pose_dim=pose_dim, n_classes=len(LABEL_MAP)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            train_epoch(model, train_loader, opt, criterion)
        val_loss, val_acc, val_f1, cm = eval_epoch(model, test_loader, criterion)
        print("LOSO test acc/f1:", val_acc, val_f1)
        results.append((test_csv, val_acc, val_f1))
    return results

# -------------------------
# Guardado dataset .npz (opcional)
# -------------------------
def export_dataset_npz(csv_path, out_path='dataset_final.npz'):
    ds = VRMultimodalDataset(csv_path)
    np.savez(out_path, ecg=ds.ecg_windows, pose=ds.pose_windows, labels=ds.labels)
    print("Guardado", out_path)

# -------------------------
# MAIN de prueba (ajusta path)
# -------------------------
if __name__ == '__main__':
    # reemplaza por tu csv real
    csv_path = 'session_01.csv'
    if os.path.exists(csv_path):
        example_train(csv_path, epochs=20, batch_size=16, lr=1e-3)
    else:
        print("Pon tu archivo CSV en same folder con nombre 'session_01.csv' o cambia csv_path.")
