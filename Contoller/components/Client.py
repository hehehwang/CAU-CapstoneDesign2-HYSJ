import socket
from time import sleep


class Client:
    def __init__(self, host="localhost", port=5689) -> None:
        self.host = host
        self.port = port
        self.socket = socket.socket()
        self.socket.settimeout(10)
        while 1:
            try:
                self.socket.connect((self.host, self.port))
                break
            except ConnectionRefusedError:
                print("Connection refused, retrying in 5 seconds...")
                sleep(5)

    def __del__(self):
        self.socket.close()

    def send(self, data: str) -> str:
        self.socket.send(data.encode())
        return self.socket.recv(4096).decode()
