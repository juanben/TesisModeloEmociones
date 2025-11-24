import serial, time

for port in [ "COM4", "COM5"]:
    try:
        ser = serial.Serial(port, 115200, timeout=2)
        print(f"Probando {port}...")
        time.sleep(2)
        data = ser.readline().decode(errors="ignore").strip()
        print(f"{port} -> {data}")
        ser.close()
    except Exception as e:
        print(f"{port} error: {e}")
