import socket
import struct
import threading
from typing import Dict, Optional


class VehicleDataClient:
    map = {
        "Position.x": 0,
        "Position.y": 1,
        "Position.z": 2,
        "Speed": 3,
        "HasFinished": 4,
    }

    def __init__(self, host: str = '127.0.0.1', port: int = 12345) -> None:
        self.host: str = host
        self.port: int = port
        self.socket: Optional[socket.socket] = None
        self.data: Dict[int, float] = {}
        self.lock: threading.Lock = threading.Lock()
        self.running: bool = False
        self.semaphore: threading.Semaphore = threading.Semaphore(0)
        self.thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            self.running = True
            print(f"OpenPlanet polaczone")
            self.thread = threading.Thread(target=self._receive_data, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"OpenPlanet blad: {e}")

    def disconnect(self) -> None:
        self.running = False
        if self.socket:
            self.socket.close()
        print("OpenPlanet rozlaczone")

    def _receive_data(self) -> None:
        if self.socket is None:
            return

        while self.running:
            try:
                frame = []
                for _ in range(len(self.map)):
                    chunk = self.socket.recv(4)
                    if not chunk:
                        raise ConnectionResetError
                    frame.append(struct.unpack('@f', chunk)[0])

                with self.lock:
                    for i, val in enumerate(frame):
                        self.data[i] = val

                self.semaphore.release()

            except ConnectionResetError:
                print("OpenPlanet kaput")
                self.disconnect()
                break

    def get_value(self, name: str) -> float:
        with self.lock:
            return self.data[self.map[name]]
