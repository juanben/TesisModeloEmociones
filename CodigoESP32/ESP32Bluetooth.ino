#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// Definiciones de UUIDs
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// Pines de tu sensor
#define ECG_PIN 36
#define LO_PLUS 34
#define LO_MINUS 35

// Variables para el Bluetooth
BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;

// Callback para manejar la conexión/desconexión
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    }
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
    }
};

// Lógica de filtro de promedio móvil
const int windowSize = 5;
int ecgBuffer[windowSize];
int bufferIndex = 0;

int movingAverage(int newValue) {
  ecgBuffer[bufferIndex] = newValue;
  bufferIndex = (bufferIndex + 1) % windowSize;

  int sum = 0;
  for (int i = 0; i < windowSize; i++) {
    sum += ecgBuffer[i];
  }
  return sum / windowSize;
}

void setup() {
  Serial.begin(115200);

  // Inicializa los pines
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  // Inicializa el buffer
  for (int i = 0; i < windowSize; i++) {
    ecgBuffer[i] = 0;
  }

  // 1. Inicializar el dispositivo BLE
  BLEDevice::init("ECG_ESP32"); 

  // 2. Crear el servidor BLE
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // 3. Crear el servicio BLE
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // 4. Crear la característica BLE
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ   |
                      BLECharacteristic::PROPERTY_WRITE  |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
  pCharacteristic->addDescriptor(new BLE2902());

  // 5. Iniciar el servicio
  pService->start();

  // 6. Configurar y empezar a publicitar
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("ESP32 BLE listo. Esperando conexión del cliente...");
}

void loop() {
  if (deviceConnected) {
    int filtered = 0;
    if (digitalRead(LO_PLUS) == 1 || digitalRead(LO_MINUS) == 1) {
      filtered = 0;
    } else {
      int ecg = analogRead(ECG_PIN);
      filtered = movingAverage(ecg);
    }

    // Convertir el valor a una cadena
    String ecgString = String(filtered);

    // Enviar el valor a través de la característica BLE
    pCharacteristic->setValue(ecgString.c_str());
    pCharacteristic->notify();

    delay(10);
  }

  // Si no hay cliente, reinicia la publicidad
  if (!deviceConnected) {
    delay(500); 
    pServer->startAdvertising();
  }
}