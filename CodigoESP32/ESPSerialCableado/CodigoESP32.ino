#define ECG_PIN 36
#define LO_PLUS 34
#define LO_MINUS 35

const int windowSize = 5;
int ecgBuffer[windowSize];
int bufferIndex = 0;

void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  // Inicializa el buffer con ceros
  for (int i = 0; i < windowSize; i++) {
    ecgBuffer[i] = 0;
  }
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
  if (digitalRead(LO_PLUS) == 1 || digitalRead(LO_MINUS) == 1) {
    Serial.println("0");
  } else {
    int ecg = analogRead(ECG_PIN);
    int filtered = movingAverage(ecg);
    Serial.println(filtered);
  }

  delay(10); // ~100Hz
}
