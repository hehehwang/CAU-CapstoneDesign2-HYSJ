import socket
from time import sleep


class Server:
    def __init__(self, host="0.0.0.0", port=5689):
        self.host = host
        self.port = port
        self.socket = socket.socket()
        self.socket.bind((self.host, self.port))
        self.conn, self.addr = self.__connect()

    def __connect(self):
        print("Waiting for connection...")
        self.socket.listen(1)
        self.socket.settimeout(5)
        while True:
            try:
                self.conn, self.addr = self.socket.accept()
                print("Connected to: ", self.addr)
                return self.conn, self.addr
            except socket.timeout:
                print("Connection timed out.")
                sleep(1)
                print("Retrying...")
                continue

    def send(self, msg: str):
        self.conn.sendall(msg.encode())

    def receive(self) -> str:
        return self.conn.recv(4).decode()

    def __del__(self):
        self.conn.close()
        self.socket.close()


def main():
    s = Server()
    while 1:
        msg = s.receive()
        print(msg)
        if msg == "exit":
            break
        s.send(msg)


if __name__ == "__main__":
    main()
