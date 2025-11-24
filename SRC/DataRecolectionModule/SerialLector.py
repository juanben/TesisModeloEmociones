import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configura el puerto correcto (ajusta COMx)
puerto = serial.Serial('COM5', 115200)

# Lista para guardar valores
valores = []

# Número de muestras a mostrar
ventana = 500

def actualizar(frame):
    while puerto.in_waiting:
        try:
            dato = puerto.readline().decode().strip()
            valor = int(dato)
            valores.append(valor)
            if len(valores) > ventana:
                valores.pop(0)
            linea.set_ydata(valores + [0]*(ventana - len(valores)))
        except:
            pass
    return linea,

fig, ax = plt.subplots()
linea, = ax.plot([0]*ventana)
ax.set_ylim(500, 3000)  # Ajusta según tu señal
ax.set_xlim(0, ventana)
ax.set_title("ECG en tiempo real")

ani = animation.FuncAnimation(fig, actualizar, interval=10)
plt.show()
