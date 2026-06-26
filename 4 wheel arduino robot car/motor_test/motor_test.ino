#include <Wire.h>
#include <Adafruit_MotorShield.h>

Adafruit_MotorShield shield = Adafruit_MotorShield();

Adafruit_DCMotor *M1 = shield.getMotor(1);  // Front-right
Adafruit_DCMotor *M2 = shield.getMotor(2);  // Front-left
Adafruit_DCMotor *M3 = shield.getMotor(3);  // Rear-left
Adafruit_DCMotor *M4 = shield.getMotor(4);  // Rear-right

const uint8_t TEST_SPEED = 220;
const unsigned long RUN_TIME_MS = 2500;
const unsigned long PAUSE_TIME_MS = 1200;

void startMotor(Adafruit_DCMotor *motor) {
  motor->setSpeed(TEST_SPEED);
  motor->run(FORWARD);
}

void stopMotor(Adafruit_DCMotor *motor) {
  motor->setSpeed(0);
  motor->run(RELEASE);
}

void stopAll() {
  stopMotor(M1);
  stopMotor(M2);
  stopMotor(M3);
  stopMotor(M4);
}

void runTest(const __FlashStringHelper *name,
             bool useM1, bool useM2, bool useM3, bool useM4) {
  Serial.println();
  Serial.print(F("TEST: "));
  Serial.println(name);

  if (useM1) startMotor(M1);
  if (useM2) startMotor(M2);
  if (useM3) startMotor(M3);
  if (useM4) startMotor(M4);

  delay(RUN_TIME_MS);

  stopAll();
  delay(PAUSE_TIME_MS);
}

void setup() {
  Serial.begin(115200);
  shield.begin();

  stopAll();

  Serial.println(F("Motor shield test starting..."));
  delay(1500);

  // Individual channels
  runTest(F("M1 only - front right"), true,  false, false, false);
  runTest(F("M2 only - front left"),  false, true,  false, false);
  runTest(F("M3 only - rear left"),   false, false, true,  false);
  runTest(F("M4 only - rear right"),  false, false, false, true);

  // Physical axle pairs
  runTest(F("Front pair: M1 + M2"), true,  true,  false, false);
  runTest(F("Rear pair: M3 + M4"),  false, false, true,  true);

  // Differential-drive sides
  runTest(F("Left side: M2 + M3"),  false, true,  true,  false);
  runTest(F("Right side: M1 + M4"), true,  false, false, true);

  // Everything
  runTest(F("ALL FOUR: M1 + M2 + M3 + M4"), true, true, true, true);

  Serial.println();
  Serial.println(F("Motor test complete."));
}

void loop() {
}