"""
Optional Arduino link. Sends the 7 lamp states as a line such as "1000011\n"
(NS R,Y,G  EW R,Y,G  PED). The state is re-sent every 0.5 s as a heartbeat;
if the Arduino hears nothing for 3 s it falls back to flashing amber,
the same fail-safe real junction controllers use.
"""
import time


class SerialLink:
    def __init__(self, port, baud=115200):
        self.ser = None
        self.last_bits = None
        self.last_sent = 0.0
        if not port:
            return
        try:
            import serial  # pyserial
            self.ser = serial.Serial(port, baud, timeout=0)
            time.sleep(2.0)  # the Arduino resets when the port opens
            print(f"[hardware] Connected to Arduino on {port}")
        except Exception as e:  # keep running in on-screen simulation
            print(f"[hardware] Could not open {port}: {e}\n[hardware] Continuing with on-screen signals only.")
            self.ser = None

    @property
    def connected(self):
        return self.ser is not None

    def send(self, lamps, now):
        if not self.ser:
            return
        bits = lamps.bits()
        if bits != self.last_bits or now - self.last_sent > 0.5:
            try:
                self.ser.write((bits + "\n").encode())
                self.last_bits, self.last_sent = bits, now
            except Exception as e:
                print(f"[hardware] Lost connection: {e}")
                self.ser = None

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


def list_ports():
    try:
        from serial.tools import list_ports as lp
        return [f"{p.device}  ({p.description})" for p in lp.comports()]
    except Exception:
        return []
