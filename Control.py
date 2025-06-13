import websocket
import threading
import time
from config import WS_IP

class WSClient:
    def __init__(self):
        self.ws = None
        self.last_sent_time = 0
        self.connected = False
        self.ws_lock = threading.Lock()

        threading.Thread(target=self.connect_ws, daemon=True).start()

    def connect_ws(self):
        while not self.connected:
            try:
                self.ws = websocket.create_connection(WS_IP)
                self.connected = True
                print("[✓] Connected to ESP8266 WebSocket")
            except Exception as e:
                print(f"[!] WebSocket retrying... {e}")
                time.sleep(2)

    def send(self, finger_data):
        if not self.connected:
            print("[!] Not connected, skipping send")
            return

        try:
            self.ws.send(str(finger_data))
            print(f"[→] Sent to ESP: {finger_data}")
            self.last_sent_time = time.time()
        except Exception as e:
            print(f"[!] WebSocket send failed: {e}")
            self.connected = False
            threading.Thread(target=self.connect_ws, daemon=True).start()
