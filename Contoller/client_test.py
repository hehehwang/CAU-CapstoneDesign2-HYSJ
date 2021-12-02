import json

from components.Client import Client
from components.ServoController import ServoController


def main():
    client = Client("0.0.0.0")
    sc = ServoController(1)
    sc.initialize()
    while 1:
        data = client.send("ready")
        if data == "exit":
            break
        print(data)
        d = json.loads(data)
        print(data)
        sc.set_angle(0, d[0])
        # sc.set_angle(0, i)


if __name__ == "__main__":
    main()
