#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include <SoftwareSerial.h>
#include <stdlib.h>
#include <ctype.h>

// ============================================================
// HC-05 Bluetooth
//
// Arduino pin 8 <- HC-05 TX
// Arduino pin 9 -> HC-05 RX
//
// Pin 12 is connected to the module's EN/KEY pin on your robot,
// but is intentionally not driven here. Since the HC-05 is already
// advertising and pairable, leave its existing EN wiring alone.
// ============================================================
const uint8_t BT_RX_PIN = 9;
const uint8_t BT_TX_PIN = 8;
const long BT_BAUD = 9600;

SoftwareSerial bluetooth(BT_RX_PIN, BT_TX_PIN);

// ============================================================
// Adafruit Motor Shield V2
// ============================================================
Adafruit_MotorShield motorShield = Adafruit_MotorShield();

Adafruit_DCMotor *M1 = motorShield.getMotor(1);  // Front-right
Adafruit_DCMotor *M2 = motorShield.getMotor(2);  // Front-left
Adafruit_DCMotor *M3 = motorShield.getMotor(3);  // Rear-left
Adafruit_DCMotor *M4 = motorShield.getMotor(4);  // Rear-right

// ============================================================
// Direction calibration
//
// Start with all false.
//
// Test: D,100,100
// If any individual wheel turns opposite the intended forward
// direction, change only that motor's value to true.
// ============================================================
const bool M1_REVERSED = false;
const bool M2_REVERSED = false;
const bool M3_REVERSED = false;
const bool M4_REVERSED = false;

// ============================================================
// Safety behavior
//
// MPPI should send a fresh D command every control cycle.
// At 10–20 Hz, 300 ms allows a few missed packets but stops
// the robot quickly if Bluetooth or the laptop process fails.
// ============================================================
const unsigned long COMMAND_TIMEOUT_MS = 5000;

unsigned long lastValidCommandMs = 0;
bool robotMoving = false;

// Bluetooth line buffer: enough for "D,-255,-255"
char commandBuffer[32];
uint8_t commandLength = 0;


// ============================================================
// Motor functions
// ============================================================

// signedPwm:
//   positive = logical forward
//   negative = logical reverse
//   zero     = release motor
void setMotor(Adafruit_DCMotor *motor, int signedPwm, bool reversed) {
  signedPwm = constrain(signedPwm, -255, 255);

  if (reversed) {
    signedPwm = -signedPwm;
  }

  if (signedPwm == 0) {
    motor->setSpeed(0);
    motor->run(RELEASE);
    return;
  }

  motor->setSpeed(abs(signedPwm));
  motor->run(signedPwm > 0 ? FORWARD : BACKWARD);
}


// Left side:  M2 front-left, M3 rear-left
// Right side: M1 front-right, M4 rear-right
void setDrive(int leftPwm, int rightPwm) {
  setMotor(M1, rightPwm, M1_REVERSED);
  setMotor(M2, leftPwm,  M2_REVERSED);
  setMotor(M3, leftPwm,  M3_REVERSED);
  setMotor(M4, rightPwm, M4_REVERSED);
}


void stopRobot() {
  setDrive(0, 0);
  robotMoving = false;
}


// ============================================================
// Command parsing
//
// Valid commands:
//
//   D,<left>,<right>
//   S
//   ?
//
// Examples:
//   D,150,150
//   D,150,90
//   D,-120,80
//   S
// ============================================================

void skipWhitespace(char *&p) {
  while (*p == ' ' || *p == '\t') {
    p++;
  }
}


bool parsePwmValue(char *&p, int &value) {
  skipWhitespace(p);

  char *endPtr;
  long parsed = strtol(p, &endPtr, 10);

  if (endPtr == p || parsed < -255 || parsed > 255) {
    return false;
  }

  p = endPtr;
  skipWhitespace(p);

  value = (int)parsed;
  return true;
}


bool parseDriveCommand(char *line, int &leftPwm, int &rightPwm) {
  char *p = line;
  skipWhitespace(p);

  if (toupper((unsigned char)*p) != 'D') {
    return false;
  }

  p++;
  skipWhitespace(p);

  if (*p != ',') {
    return false;
  }
  p++;

  if (!parsePwmValue(p, leftPwm)) {
    return false;
  }

  if (*p != ',') {
    return false;
  }
  p++;

  if (!parsePwmValue(p, rightPwm)) {
    return false;
  }

  // Reject extra junk after the second PWM number.
  return (*p == '\0');
}


void sendHelp() {
  bluetooth.println(F("D,<left>,<right> where each is -255..255"));
  bluetooth.println(F("Examples: D,150,150  D,150,80  D,0,0"));
  bluetooth.println(F("S = stop"));
}


void handleCommand(char *line) {
  char *p = line;
  skipWhitespace(p);

  if (*p == '\0') {
    return;
  }

  // Stop command
  if ((toupper((unsigned char)*p) == 'S') && *(p + 1) == '\0') {
    stopRobot();
    bluetooth.println(F("STOP"));
    return;
  }

  // Help command
  if (*p == '?') {
    sendHelp();
    return;
  }

  int leftPwm;
  int rightPwm;

  if (parseDriveCommand(line, leftPwm, rightPwm)) {
    setDrive(leftPwm, rightPwm);

    robotMoving = (leftPwm != 0 || rightPwm != 0);
    lastValidCommandMs = millis();

    // Intentionally no Bluetooth reply for D commands.
    // This keeps the MPPI command stream simple and one-way.
    return;
  }

  bluetooth.println(F("ERR: use D,<left>,<right> or S"));
}


void readBluetoothCommands() {
  while (bluetooth.available()) {
    char c = (char)bluetooth.read();

    // Accept either CR, LF, or CRLF line endings.
    if (c == '\r' || c == '\n') {
      if (commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        handleCommand(commandBuffer);
        commandLength = 0;
      }
      continue;
    }

    // Prevent buffer overflow from an accidental long message.
    if (commandLength < sizeof(commandBuffer) - 1) {
      commandBuffer[commandLength++] = c;
    } else {
      commandLength = 0;
      bluetooth.println(F("ERR: command too long"));
    }
  }
}


void setup() {
  Serial.begin(115200);
  bluetooth.begin(BT_BAUD);

  motorShield.begin();
  stopRobot();

  Serial.println(F("Robot ready."));
  Serial.println(F("Bluetooth protocol: D,<left PWM>,<right PWM>"));
  bluetooth.println(F("Robot ready. Send ? for help."));
}


void loop() {
  readBluetoothCommands();

  // Dead-man safety stop if the command stream disappears.
  if (robotMoving &&
      (millis() - lastValidCommandMs > COMMAND_TIMEOUT_MS)) {
    stopRobot();
    bluetooth.println(F("Safety stop"));
  }
}