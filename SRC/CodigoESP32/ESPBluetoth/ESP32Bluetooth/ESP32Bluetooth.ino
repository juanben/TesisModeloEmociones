#include "BluetoothSerial.h"

BluetoothSerial SerialBT;  // Objeto Bluetooth Serial

#define ECG_PIN 36
#define LO_PLUS 34
#define LO_MINUS 35

const int windowSize = 5;
int ecgBuffer[windowSize];
int bufferIndex = 0;

void setup() {
  Serial.begin(115200);             // Serial por USB
  SerialBT.begin("ECG_ESP32");      // Nombre del dispositivo Bluetooth

  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  // Inicializa el buffer con ceros
  for (int i = 0; i < windowSize; i++) {
    ecgBuffer[i] = 0;
  }

  Serial.println("ECG listo. Conéctate por Bluetooth con el nombre: ECG_ESP32");
}

int movingAverage(int newValue) {
  ecgBuffer[bufferIndex] = newValue;
  bufferIndex = (bufferIndex + 1) % windowSize;

  int sum = 0;
  for (int i = 0; i < windowSize; i++) {
    sum += ecgBuffer[i];
  }
  return sum / windowSize;
}

void loop() {
  int value = 0;

  if (digitalRead(LO_PLUS) == 1 || digitalRead(LO_MINUS) == 1) {
    value = 0;
  } else {
    int ecg = analogRead(ECG_PIN);
    value = movingAverage(ecg);
  }

  // Enviar por USB Serial y por Bluetooth Serial
  Serial.println(value);
  SerialBT.println(value);

  delay(10); // ~100 Hz
}
