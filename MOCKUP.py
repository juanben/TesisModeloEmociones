import os
import time
import csv
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw

# Visual Config
WINDOW_SECONDS = 3.0
FPS_TARGET = 10.4
TIMESTEPS = int(round(WINDOW_SECONDS * FPS_TARGET))  # 31
SAVE_DIR = "RT_Logs"


class DummyCamera:
    """Simulates a camera feed by generating test frames."""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

    def read_frame(self, text_overlay=""):
        # Create a dark canvas representing video
        img = Image.new("RGB", (self.width, self.height), color=(30, 30, 35))
        draw = ImageDraw.Draw(img)

        # Draw background grid
        for x in range(0, self.width, 40):
            draw.line([(x, 0), (x, self.height)], fill=(45, 45, 50), width=1)
        for y in range(0, self.height, 40):
            draw.line([(0, y), (self.width, y)], fill=(45, 45, 50), width=1)

        # Static bounding box frame
        cx, cy = self.width // 2, self.height // 2
        draw.rectangle([cx - 100, cy - 100, cx + 100, cy + 100], outline=(0, 200, 255), width=2)
        draw.text((cx - 50, cy - 10), "[ MOCK CAMERA ]", fill=(200, 200, 200))

        # Overlay text on frame
        if text_overlay:
            draw.text((20, 20), text_overlay, fill=(0, 255, 100))

        return img


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VR Emotion - RealTime")
        self.root.geometry("1100x650")

        os.makedirs(SAVE_DIR, exist_ok=True)

        # Video simulator
        self.dummy_cam = DummyCamera()

        # UI State
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None

        # ---------------- HARDCODED VALUES ----------------
        self.FIXED_EMOTION = "Neutral"
        self.FIXED_CONFIDENCE = "0.73"
        self.FIXED_BUFFER = f"{TIMESTEPS}/{TIMESTEPS}"
        self.FIXED_PROBS = {
            "Neutral": "0.733",
            "Fear": "0.190",
            "Anger": "0.022",
            "Joy": "0.054"
        }

        # ---------------- UI ----------------
        self._build_ui()

        # Start main render loop
        self._update_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Left: Video panel
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.video_label = ttk.Label(left)
        self.video_label.pack(fill="both", expand=True)

        # Right: Controls panel
        right = ttk.Frame(main, width=320)
        right.pack(side="right", fill="y", padx=(10, 0))

        ttk.Label(right, text="Control Panel", font=("Segoe UI", 14, "bold")).pack(pady=(0, 10))

        # Filename input
        ttk.Label(right, text="Filename (without .csv):").pack(anchor="w")
        self.filename_var = tk.StringVar(value=f"session_{time.strftime('%Y%m%d_%H%M%S')}")
        ttk.Entry(right, textvariable=self.filename_var).pack(fill="x", pady=(0, 10))

        # Action buttons
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=5)

        self.btn_start = ttk.Button(btns, text="▶ Start", command=self.start_recording)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_stop = ttk.Button(btns, text="■ Stop", command=self.stop_recording, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True)

        ttk.Separator(right).pack(fill="x", pady=15)

        # Status indicator
        self.status_var = tk.StringVar(value="Ready. (Not recording)")
        ttk.Label(right, textvariable=self.status_var, wraplength=300).pack(anchor="w")

        ttk.Separator(right).pack(fill="x", pady=15)

        # Live metrics (valores quemados)
        self.emotion_var = tk.StringVar(value=f"Emotion: {self.FIXED_EMOTION}")
        self.conf_var = tk.StringVar(value=f"Confidence: {self.FIXED_CONFIDENCE}")
        self.buf_var = tk.StringVar(value=f"Buffer: {self.FIXED_BUFFER}")

        ttk.Label(right, textvariable=self.emotion_var, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(5, 0))
        ttk.Label(right, textvariable=self.conf_var).pack(anchor="w")
        ttk.Label(right, textvariable=self.buf_var).pack(anchor="w")

        ttk.Separator(right).pack(fill="x", pady=15)

        # Probabilities table (valores quemados)
        ttk.Label(right, text="Probabilities (smoothed):", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))
        for lab, val in self.FIXED_PROBS.items():
            ttk.Label(right, text=f"{lab}: {val}").pack(anchor="w")

    def start_recording(self):
        name = self.filename_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a valid filename.")
            return

        path = os.path.join(SAVE_DIR, f"{name}.csv")
        if os.path.exists(path):
            if not messagebox.askyesno("Overwrite", f"File '{path}' already exists. Overwrite?"):
                return

        self.csv_file = open(path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "label", "conf", "p_neutral", "p_fear", "p_anger", "p_joy"])

        self.is_recording = True
        self.status_var.set(f"Recording to: {path}")
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

        self.status_var.set("Finished. (Not recording)")

    def _update_frame(self):
        # Texto quemado sobre el video
        overlay_text = f"Emotion: {self.FIXED_EMOTION} | Confidence: {self.FIXED_CONFIDENCE}"
        img = self.dummy_cam.read_frame(overlay_text)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Si está grabando, guarda el registro estático
        if self.is_recording and self.csv_writer:
            ts = time.time()
            self.csv_writer.writerow([
                ts, self.FIXED_EMOTION, self.FIXED_CONFIDENCE,
                self.FIXED_PROBS["Neutral"], self.FIXED_PROBS["Fear"],
                self.FIXED_PROBS["Anger"], self.FIXED_PROBS["Joy"]
            ])

        # Render loop
        self.root.after(33, self._update_frame)

    def on_close(self):
        try:
            self.stop_recording()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()