/*
  SignalSync - Arduino traffic signal (Uno / Nano)

  Receives lines of 7 characters from the laptop, e.g. "1000011"
     NS: Red Yellow Green | EW: Red Yellow Green | Pedestrian WALK
  and lights the LEDs to match.

  Fail-safe: if nothing arrives for 3 seconds (laptop closed, cable pulled,
  app crashed) both roads switch to FLASHING AMBER, like a real junction
  controller does on a fault.

  Wiring (each LED: Arduino pin -> 220 ohm resistor -> LED long leg, short leg -> GND)
     Pin 2  NS Red      Pin 5  EW Red      Pin 8  Pedestrian (white/blue)
     Pin 3  NS Yellow   Pin 6  EW Yellow
     Pin 4  NS Green    Pin 7  EW Green
*/

const int PINS[7] = {2, 3, 4, 5, 6, 7, 8};
const unsigned long TIMEOUT_MS = 3000;

char buf[16];
int len = 0;
unsigned long lastMsg = 0;

void setAll(const char *bits) {
  for (int i = 0; i < 7; i++) {
    digitalWrite(PINS[i], bits[i] == '1' ? HIGH : LOW);
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 7; i++) pinMode(PINS[i], OUTPUT);
  // power-on lamp test: all LEDs on for half a second
  setAll("1111111");
  delay(500);
  setAll("0000000");
  lastMsg = 0;  // start in fail-safe until the laptop speaks
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (len == 7) {
        buf[7] = '\0';
        bool valid = true;
        for (int i = 0; i < 7; i++) if (buf[i] != '0' && buf[i] != '1') valid = false;
        // never allow two greens, or WALK with a green, even if a bad message arrives
        if (buf[2] == '1' && buf[5] == '1') valid = false;
        if (buf[6] == '1' && (buf[2] == '1' || buf[5] == '1')) valid = false;
        if (valid) {
          setAll(buf);
          lastMsg = millis();
        }
      }
      len = 0;
    } else if (len < 15) {
      buf[len++] = c;
    }
  }

  if (lastMsg == 0 || millis() - lastMsg > TIMEOUT_MS) {
    bool on = (millis() / 500) % 2 == 0;      // flashing amber, 1 Hz
    setAll(on ? "0100100" : "0000000");
  }
}
