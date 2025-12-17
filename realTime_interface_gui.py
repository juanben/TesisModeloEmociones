import os
import time
import csv
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp

from realTime_core import ECGSerialReader, RealTimeEmotionRunner, LABELS


# ==============================
# CONFIG (ajusta rutas/puertos)
# ==============================
MODEL_PATH  = "Modelo/model_attention_LOSO_10H.keras"        # tu modelo final
SCALER_PATH = "Modelo/scaler_attention_LOSO_10H.npz"   # tu scaler mean/std

COM_PORT  = "COM7"
BAUD_RATE = 115200
CAM_INDEX = 0

WINDOW_SECONDS = 3.0
FPS_TARGET = 10.4
TIMESTEPS = int(round(WINDOW_SECONDS * FPS_TARGET))  # ~31
OVERLAP_RATIO = 0.5
STEP = max(1, int(round(TIMESTEPS * (1 - OVERLAP_RATIO))))  # ~15

SAVE_DIR = "RegistrosRT"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VR Emotion - RealTime")
        self.root.geometry("1100x650")

        os.makedirs(SAVE_DIR, exist_ok=True)

        # runner/model
        self.runner = RealTimeEmotionRunner(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            timesteps=TIMESTEPS,
            step=STEP,
            ecg_median_k=5,
            pose_smooth_k=3,
            pred_smooth_k=5,
        )
        self.runner.load()

        # serial reader
        self.ecg_reader = ECGSerialReader(COM_PORT, BAUD_RATE)
        self.ecg_reader.start()

        # camera + mediapipe
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara.")

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # UI state
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.last_probs = None

        # ---------------- UI ----------------
        self._build_ui()

        # loop
        self._update_frame()

        # close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # left: video
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.video_label = ttk.Label(left)
        self.video_label.pack(fill="both", expand=True)

        # right: controls
        right = ttk.Frame(main, width=320)
        right.pack(side="right", fill="y")

        ttk.Label(right, text="Control", font=("Segoe UI", 14, "bold")).pack(pady=(0, 10))

        # filename
        ttk.Label(right, text="Nombre de archivo (sin .csv):").pack(anchor="w")
        self.filename_var = tk.StringVar(value=f"session_{time.strftime('%Y%m%d_%H%M%S')}")
        ttk.Entry(right, textvariable=self.filename_var).pack(fill="x", pady=(0, 10))

        # buttons
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=5)

        self.btn_start = ttk.Button(btns, text="▶ Iniciar", command=self.start_recording)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_stop = ttk.Button(btns, text="■ Finalizar", command=self.stop_recording, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True)

        ttk.Separator(right).pack(fill="x", pady=15)

        # status
        self.status_var = tk.StringVar(value="Listo. (No grabando)")
        ttk.Label(right, textvariable=self.status_var, wraplength=300).pack(anchor="w")

        ttk.Separator(right).pack(fill="x", pady=15)

        # live metrics
        self.emotion_var = tk.StringVar(value="Emoción: ...")
        self.conf_var = tk.StringVar(value="Confianza: 0.00")
        self.buf_var = tk.StringVar(value=f"Buffer: 0/{TIMESTEPS}")

        ttk.Label(right, textvariable=self.emotion_var, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(5,0))
        ttk.Label(right, textvariable=self.conf_var).pack(anchor="w")
        ttk.Label(right, textvariable=self.buf_var).pack(anchor="w")

        ttk.Separator(right).pack(fill="x", pady=15)

        # probs table
        ttk.Label(right, text="Probabilidades (suavizadas):", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))
        self.prob_labels = {}
        for lab in LABELS:
            v = tk.StringVar(value=f"{lab}: 0.000")
            self.prob_labels[lab] = v
            ttk.Label(right, textvariable=v).pack(anchor="w")

    def start_recording(self):
        name = self.filename_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Pon un nombre de archivo.")
            return

        path = os.path.join(SAVE_DIR, f"{name}.csv")
        if os.path.exists(path):
            if not messagebox.askyesno("Sobrescribir", f"Ya existe {path}. ¿Sobrescribir?"):
                return

        self.csv_file = open(path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "label", "conf", "p_neutro", "p_miedo", "p_ira", "p_alegria"])

        self.is_recording = True
        self.status_var.set(f"Grabando en: {path}")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

    def stop_recording(self):
        self.is_recording = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

        if self.csv_file:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass
        self.csv_file = None
        self.csv_writer = None

        self.status_var.set("Finalizado. (No grabando)")

    def _update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Error leyendo cámara.")
            self.root.after(30, self._update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(frame_rgb)

        ecg_raw, _ = self.ecg_reader.get_latest()

        landmarks = None
        if res.pose_landmarks is not None:
            lms = res.pose_landmarks.landmark
            landmarks = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in range(33)], dtype=np.float32)

        if ecg_raw is not None:
            out = self.runner.update(float(ecg_raw), landmarks)
            self.buf_var.set(f"Buffer: {out.get('buffer_len',0)}/{TIMESTEPS}")

            if out.get("ready", False):
                self.emotion_var.set(f"Emoción: {out['label']}")
                self.conf_var.set(f"Confianza: {out['conf']:.2f}")

                if out.get("probs") is not None:
                    probs = out["probs"]
                    self.last_probs = probs
                    for i, lab in enumerate(LABELS):
                        self.prob_labels[lab].set(f"{lab}: {probs[i]:.3f}")

                # grabar SOLO cuando hay predicción nueva
                if self.is_recording and out.get("did_predict", False) and (out.get("probs") is not None):
                    ts = time.time()
                    probs = out["probs"]
                    self.csv_writer.writerow([ts, out["label"], f"{out['conf']:.4f}",
                                              f"{probs[0]:.6f}", f"{probs[1]:.6f}", f"{probs[2]:.6f}", f"{probs[3]:.6f}"])
        else:
            # ECG no listo
            self.emotion_var.set("Emoción: (esperando ECG)")
            self.conf_var.set("Confianza: --")

        # overlay en la imagen (limpio)
        overlay_text = self.emotion_var.get() + " | " + self.conf_var.get()
        cv2.putText(frame, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        # mostrar en Tkinter
        frame_rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb_show)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(15, self._update_frame)

    def on_close(self):
        try:
            self.stop_recording()
        except Exception:
            pass

        try:
            self.ecg_reader.stop()
        except Exception:
            pass

        try:
            self.cap.release()
        except Exception:
            pass

        self.root.destroy()


def main():
    root = tk.Tk()
    # estilo
    try:
        from tkinter import font
    except Exception:
        pass

    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
