import os
import time
import threading
from collections import deque

import numpy as np
import serial
import tensorflow as tf

# IMPORTANTE: tu capa custom debe existir y estar registrada/importada
from SRC.LSTM.model_lstm_atention import ReduceSumTime

LABELS = ["neutro", "miedo", "ira", "alegria"]


class ECGSerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self._stop = threading.Event()

        self.lock = threading.Lock()
        self.latest_ecg = None
        self.latest_ts = None
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(1.0)
            print(f"✔ Serial conectado: {self.port} @ {self.baud}")
        except Exception as e:
            print(f"❌ Error abriendo Serial: {e}")
            return

        while not self._stop.is_set():
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue

                # intenta extraer un número
                parts = "".join([c if (c.isdigit() or c in "-.") else " " for c in line]).split()
                if not parts:
                    continue
                val = float(parts[-1])

                with self.lock:
                    self.latest_ecg = val
                    self.latest_ts = time.time()

            except Exception:
                continue

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass

    def stop(self):
        self._stop.set()

    def get_latest(self):
        with self.lock:
            return self.latest_ecg, self.latest_ts


def normalize_pose_frame(landmarks_xyz: np.ndarray) -> np.ndarray:
    """
    landmarks_xyz: (33,3) [x,y,z] de MediaPipe.
    Normalización: rotación + traslación a pelvis + escala por distancia caderas.
    """
    lm = landmarks_xyz.astype(np.float32).copy()

    shoulder = (lm[11, :2] + lm[12, :2]) / 2.0
    pelvis   = (lm[23, :2] + lm[24, :2]) / 2.0

    vx, vy = (shoulder - pelvis)
    angle = np.arctan2(vy, vx)

    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    # centrar a pelvis
    lm[:, 0] -= pelvis[0]
    lm[:, 1] -= pelvis[1]

    # rotar x,y
    x = lm[:, 0].copy()
    y = lm[:, 1].copy()
    lm[:, 0] = x * cos_a - y * sin_a
    lm[:, 1] = x * sin_a + y * cos_a

    hip_dist = float(np.sqrt((lm[23, 0] - lm[24, 0])**2 + (lm[23, 1] - lm[24, 1])**2))
    if hip_dist == 0:
        hip_dist = 1e-6

    lm[:, 0] /= hip_dist
    lm[:, 1] /= hip_dist
    lm[:, 2] /= hip_dist

    return lm


def flatten_features_103(ecg_norm, ecg_diff, ecg_speed, ecg_energy, landmarks_norm_33x3):
    """
    Orden EXACTO igual a crear_ventanas:
    [ecg_norm, ecg_diff, ecg_speed, ecg_energy, 0_x,0_y,0_z, ..., 32_z]
    """
    feat = np.empty((103,), dtype=np.float32)
    feat[0] = float(ecg_norm)
    feat[1] = float(ecg_diff)
    feat[2] = float(ecg_speed)
    feat[3] = float(ecg_energy)

    idx = 4
    for i in range(33):
        feat[idx:idx+3] = landmarks_norm_33x3[i]
        idx += 3
    return feat


class RealTimeEmotionRunner:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        timesteps: int,
        step: int,
        ecg_median_k: int = 5,
        pose_smooth_k: int = 3,
        pred_smooth_k: int = 5,
        labels=LABELS
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.timesteps = timesteps
        self.step = step
        self.labels = labels

        self.ecg_median_k = ecg_median_k
        self.pose_smooth_k = pose_smooth_k
        self.pred_smooth_k = pred_smooth_k

        self.model = None
        self.mean = None
        self.std = None

        self.window_buf = deque(maxlen=timesteps)     # (103,)
        self.ecg_buf = deque(maxlen=ecg_median_k)     # raw
        self.pose_buf = deque(maxlen=pose_smooth_k)   # (33,3)
        self.pred_buf = deque(maxlen=pred_smooth_k)   # (4,)

        self.prev_ecg_norm = None
        self.frame_count = 0

        self.current_label = "..."
        self.current_conf = 0.0
        self.last_pred_time = 0.0

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No existe model_path: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"No existe scaler_path: {self.scaler_path}")

        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
            custom_objects={"ReduceSumTime": ReduceSumTime},
        )

        scaler = np.load(self.scaler_path)
        self.mean = float(scaler["mean"])
        self.std = float(scaler["std"])
        if self.std == 0 or np.isnan(self.std):
            self.std = 1.0

        print(f"✔ Modelo cargado: {self.model_path}")
        print(f"✔ Scaler cargado: mean={self.mean:.6f}, std={self.std:.6f}")
        print(f"✔ Timesteps={self.timesteps}, STEP={self.step}")

    def update(self, ecg_raw: float, landmarks_33x3: np.ndarray | None):
        if self.model is None:
            raise RuntimeError("Runner no cargado. Llama a runner.load()")

        # ECG smooth (mediana)
        self.ecg_buf.append(ecg_raw)
        ecg_smooth = float(np.median(np.array(self.ecg_buf, dtype=np.float32)))

        # ecg_norm (global scaler)
        ecg_norm = (ecg_smooth - self.mean) / self.std

        # derivadas (online) igual a crear_ventanas
        if self.prev_ecg_norm is None:
            ecg_diff = 0.0
        else:
            ecg_diff = float(ecg_norm - self.prev_ecg_norm)
        ecg_speed = abs(ecg_diff)
        ecg_energy = float(ecg_norm ** 2)
        self.prev_ecg_norm = float(ecg_norm)

        # pose smoothing
        if landmarks_33x3 is not None:
            lm_norm = normalize_pose_frame(landmarks_33x3)
            self.pose_buf.append(lm_norm)

        if len(self.pose_buf) == 0:
            return {
                "ready": False,
                "msg": "Pose no detectada",
                "label": self.current_label,
                "conf": self.current_conf,
                "ecg_raw": ecg_raw,
                "ecg_smooth": ecg_smooth,
            }

        lm_smooth = np.mean(np.stack(self.pose_buf, axis=0), axis=0)

        feat = flatten_features_103(ecg_norm, ecg_diff, ecg_speed, ecg_energy, lm_smooth)
        self.window_buf.append(feat)
        self.frame_count += 1

        did_predict = False
        probs_smooth = None

        if len(self.window_buf) == self.timesteps and (self.frame_count % self.step == 0):
            X = np.stack(self.window_buf, axis=0)[None, :, :]  # (1,T,103)
            probs = self.model.predict(X, verbose=0)[0]        # (4,)
            self.pred_buf.append(probs)

            probs_smooth = np.mean(np.stack(self.pred_buf, axis=0), axis=0)
            cls = int(np.argmax(probs_smooth))
            self.current_label = self.labels[cls]
            self.current_conf = float(probs_smooth[cls])
            self.last_pred_time = time.time()
            did_predict = True

        return {
            "ready": True,
            "msg": "",
            "label": self.current_label,
            "conf": self.current_conf,
            "did_predict": did_predict,
            "probs": probs_smooth,  # (4,) o None
            "ecg_raw": ecg_raw,
            "ecg_smooth": ecg_smooth,
            "buffer_len": len(self.window_buf),
        }
