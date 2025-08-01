import cv2
import mediapipe as mp
import serial
import time
import csv
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Intentar abrir puerto serial
try:
    puerto = serial.Serial('COM5', 115200, timeout=1)
except serial.SerialException as e:
    print(f"Error al abrir el puerto serial: {e}")
    exit()

# Inicializar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Variables globales
grabando = False
ecg_valor = 0
valores_ecg = [0] * 500
csv_writer = None
csv_file = None
video_thread = None

# GUI
ventana = tk.Tk()
ventana.title("Adquisición Multimodal")
ventana.geometry("1200x700")

# Canvas de video
lbl_video = tk.Label(ventana)
lbl_video.grid(row=0, column=0, padx=10, pady=10)

# ECG - Matplotlib
fig, ax = plt.subplots(figsize=(5, 3))
linea, = ax.plot(valores_ecg)
ax.set_ylim(500, 3000)
ax.set_xlim(0, 500)
canvas_ecg = FigureCanvasTkAgg(fig, master=ventana)
canvas_ecg.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

# Función para actualizar gráfica ECG
def actualizar_ecg():
    global valores_ecg, ecg_valor
    if grabando and puerto.in_waiting:
        try:
            linea_serial = puerto.readline().decode().strip()
            valor = int(linea_serial)
            if 300 < valor < 3000:
                ecg_valor = valor
                valores_ecg.append(ecg_valor)
                valores_ecg = valores_ecg[-500:]
                linea.set_ydata(valores_ecg)
                canvas_ecg.draw()
        except:
            pass
    if grabando:
        ventana.after(10, actualizar_ecg)

# Función para capturar y mostrar cámara en hilo separado
def video_loop():
    global grabando, ecg_valor, csv_writer
    while grabando:
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

        # Guardar en CSV
        if csv_writer:
            csv_writer.writerow([timestamp, ecg_valor] + pose_data)

        # Mostrar imagen en Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)

        time.sleep(0.01)  # Para no saturar CPU

# Función iniciar
def iniciar():
    global grabando, csv_writer, csv_file, video_thread
    if not grabando:
        grabando = True
        csv_file = open("datos_multimodal.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        headers = ['timestamp', 'ecg'] + [f'{i}_{c}' for i in range(33) for c in ['x','y','z']]
        csv_writer.writerow(headers)
        actualizar_ecg()
        video_thread = threading.Thread(target=video_loop)
        video_thread.daemon = True
        video_thread.start()
        btn_iniciar.config(state="disabled")
        btn_detener.config(state="normal")

# Función detener
def detener():
    global grabando, csv_file
    grabando = False
    if csv_file:
        csv_file.close()
        csv_file = None
        print("Grabación finalizada")
    btn_iniciar.config(state="normal")
    btn_detener.config(state="disabled")

# Manejo de cierre de ventana
def al_cerrar():
    detener()
    ventana.destroy()

ventana.protocol("WM_DELETE_WINDOW", al_cerrar)

# Botones
btn_iniciar = ttk.Button(ventana, text="Iniciar", command=iniciar)
btn_iniciar.grid(row=1, column=0, pady=10)

btn_detener = ttk.Button(ventana, text="Detener", command=detener, state="disabled")
btn_detener.grid(row=1, column=1, pady=10)

# Ejecutar GUI
ventana.mainloop()

# Limpieza final
cap.release()
puerto.close()