import cv2
import mediapipe as mp
import asyncio
import bleak
import csv
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración BLE
# Reemplaza 'DIRECCION_MAC_DEL_ESP32' con la dirección MAC real de tu ESP32 o su nombre
ESP32_DEVICE_MAC = '01:23:45:67:89:AB'
# Reemplaza 'UUID_DEL_SERVICIO' y 'UUID_DE_LA_CARACTERISTICA' con los UUIDs que configuraste en tu ESP32
SERVICE_UUID = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'
CHARACTERISTIC_UUID = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

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
# Variables para BLE
client = None
loop = None

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

# Callback para la notificación de datos BLE
def notification_handler(sender, data):
    global ecg_valor, valores_ecg
    try:
        # Decodifica los datos recibidos del ESP32 (el formato puede variar)
        valor_str = data.decode('utf-8').strip()
        valor = int(valor_str)
        if 300 < valor < 3000:
            ecg_valor = valor
            valores_ecg.append(ecg_valor)
            valores_ecg = valores_ecg[-500:]
            ventana.event_generate("<<ECG_Updated>>") # Generar un evento para actualizar la GUI
    except (UnicodeDecodeError, ValueError):
        pass

# Función para actualizar gráfica ECG en la GUI
def actualizar_ecg_gui(event=None):
    global linea, valores_ecg
    linea.set_ydata(valores_ecg)
    canvas_ecg.draw_idle()

# Función asíncrona para manejar la conexión y notificaciones BLE
async def run_ble_client():
    global client, grabando, loop
    try:
        print(f"Conectando a {ESP32_DEVICE_MAC}...")
        async with bleak.BleakClient(ESP32_DEVICE_MAC, timeout=20.0) as client:
            print("Conectado!")
            # Iniciar notificaciones para la característica de ECG
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
            while grabando:
                await asyncio.sleep(0.1) # Esperar un poco para no saturar
            # Detener notificaciones al finalizar
            await client.stop_notify(CHARACTERISTIC_UUID)
    except Exception as e:
        print(f"Error en la conexión BLE: {e}")
        detener()
    finally:
        print("Desconectando cliente BLE.")
        if client and client.is_connected:
            await client.disconnect()

# Función para iniciar la conexión BLE en un hilo separado
def ble_thread_start():
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_ble_client())

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

        if csv_writer:
            csv_writer.writerow([timestamp, ecg_valor] + pose_data)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)

        time.sleep(0.01)

# Función iniciar
def iniciar():
    global grabando, csv_writer, csv_file, video_thread
    if not grabando:
        grabando = True
        csv_file = open("datos_multimodal.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        headers = ['timestamp', 'ecg'] + [f'{i}_{c}' for i in range(33) for c in ['x','y','z']]
        csv_writer.writerow(headers)
        
        # Conectar el evento para actualizar la gráfica
        ventana.bind("<<ECG_Updated>>", actualizar_ecg_gui)
        
        # Iniciar el hilo de la conexión BLE
        ble_thread = threading.Thread(target=ble_thread_start)
        ble_thread.daemon = True
        ble_thread.start()
        
        # Iniciar el hilo de video
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
    if loop and loop.is_running():
        loop.stop()
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
print("Recursos liberados")