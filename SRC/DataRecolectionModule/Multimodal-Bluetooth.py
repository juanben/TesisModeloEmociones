import cv2
import mediapipe as mp
import serial
import csv
import threading
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# ------------------- CONFIGURACIÓN -------------------
BT_PORT = "COM4"   # <-- Cambia según tu caso
BAUD_RATE = 115200

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Variables globales
grabando = False
ecg_valor = 0
valores_ecg = [0] * 500
csv_writer = None
csv_file = None
video_thread = None
serial_thread = None
ser = None
cap = None
nombre_csv = None

# ------------------- FUNCIONES -------------------

def serial_loop():
    global ecg_valor, valores_ecg, grabando, ser
    while grabando and ser:
        try:
            line = ser.readline().decode().strip()
            if line.isdigit():
                valor = int(line)
                if 300 < valor < 3000:
                    ecg_valor = valor
                    valores_ecg.append(ecg_valor)
                    valores_ecg = valores_ecg[-500:]
                    ventana.event_generate("<<ECG_Updated>>")
        except Exception as e:
            print("Error lectura serial:", e)
            break

def actualizar_ecg_gui(event=None):
    global linea, valores_ecg
    linea.set_ydata(valores_ecg)
    canvas_ecg.draw_idle()

def video_loop():
    global grabando, ecg_valor, csv_writer, cap
    while grabando:
        if not cap or not cap.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        timestamp = time.time()

        pose_data = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                pose_data.extend([lm.x, lm.y, lm.z])
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            pose_data = [0] * (33 * 3)

        if csv_writer:
            csv_writer.writerow([timestamp, ecg_valor] + pose_data)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)

        time.sleep(0.01)

def iniciar():
    global grabando, csv_writer, csv_file, video_thread, serial_thread, ser, cap, nombre_csv

    if not grabando:
        # Pedir nombre CSV
        nombre_csv = simpledialog.askstring("Nombre CSV", "Ingrese el nombre del archivo CSV:")
        if not nombre_csv:
            return
        if not nombre_csv.endswith(".csv"):
            nombre_csv += ".csv"

        # Intentar abrir puerto serial
        try:
            ser = serial.Serial(BT_PORT, BAUD_RATE, timeout=1)
            print("Conectado a", BT_PORT)
        except Exception as e:
            ser = None
            messagebox.showwarning("Advertencia", f"No se pudo conectar al puerto {BT_PORT}.\nError: {e}")

        # Intentar abrir cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None
            messagebox.showwarning("Advertencia", "No se detectó la cámara.")

        if not ser and not cap:
            messagebox.showwarning("Error", "No hay dispositivos conectados. No se puede iniciar la grabación.")
            return

        # Crear CSV
        grabando = True
        try:
            csv_file = open(nombre_csv, "w", newline='')
            csv_writer = csv.writer(csv_file)
            headers = ['timestamp', 'ecg'] + [f'{i}_{c}' for i in range(33) for c in ['x','y','z']]
            csv_writer.writerow(headers)
        except Exception as e:
            messagebox.showwarning("Error", f"No se pudo crear el CSV.\nError: {e}")
            grabando = False
            return

        # Iniciar hilos
        ventana.bind("<<ECG_Updated>>", actualizar_ecg_gui)
        serial_thread = threading.Thread(target=serial_loop, daemon=True)
        serial_thread.start()
        video_thread = threading.Thread(target=video_loop, daemon=True)
        video_thread.start()

        btn_iniciar.config(state="disabled")
        btn_detener.config(state="normal")

def detener():
    global grabando, csv_file, ser, cap, nombre_csv, csv_writer
    grabando = False

    # Cerrar serial y cámara
    if ser:
        ser.close()
        ser = None
    if cap:
        cap.release()
        cap = None

    # Cerrar archivo CSV
    if csv_file:
        csv_file.close()
        csv_file = None
        csv_writer = None
        print(f"Grabación finalizada. Archivo guardado como: {nombre_csv}")

    btn_iniciar.config(state="normal")
    btn_detener.config(state="disabled")

def al_cerrar():
    detener()
    ventana.destroy()

# ------------------- INTERFAZ -------------------

ventana = tk.Tk()
ventana.title("Adquisición Multimodal")
ventana.geometry("1200x700")

# Canvas de video
lbl_video = tk.Label(ventana)
lbl_video.grid(row=0, column=0, padx=10, pady=10)

# ECG - Matplotlib
fig, ax = plt.subplots(figsize=(5, 3))
linea, = ax.plot(valores_ecg)
ax.set_ylim(500, 5000)
ax.set_xlim(0, 500)
canvas_ecg = FigureCanvasTkAgg(fig, master=ventana)
canvas_ecg.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

# Botones
btn_iniciar = ttk.Button(ventana, text="Iniciar", command=iniciar)
btn_iniciar.grid(row=1, column=0, pady=10)

btn_detener = ttk.Button(ventana, text="Detener", command=detener, state="disabled")
btn_detener.grid(row=1, column=1, pady=10)

ventana.protocol("WM_DELETE_WINDOW", al_cerrar)

# Ejecutar GUI
ventana.mainloop()

print("Recursos liberados")
