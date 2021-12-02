from time import sleep

import busio
from adafruit_motor.servo import Servo
from adafruit_pca9685 import PCA9685
from board import SCL, SDA


class ServoController:
    i2c = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c)
    pca.frequency = 50
    MG996R = {"min_pulse": 700, "max_pulse": 2400}

    def __init__(self, number_of_servo: int) -> None:
        self.__servos = [self.__getMG996R(i) for i in range(number_of_servo)]
        self.__angles = [0 for i in range(number_of_servo)]
        self.initialize()

    def test_servo(self, servo_number: int) -> None:
        for a in range(0, 181, 10):
            self.set_angle(servo_number, a)
            sleep(0.1)
        sleep(1)
        self.set_angle(servo_number, 0)

    def set_angle(self, servo_number: int, angle: int) -> None:
        if angle < 0 or angle > 180:
            raise ValueError("Angle must be between 0 and 180")
        elif servo_number < 0 or servo_number > len(self.__servos):
            raise ValueError(
                "Servo number must be between 0 and {}".format(len(self.__servos))
            )

        self.__servos[servo_number].angle = angle
        self.__angles[servo_number] = angle

    def get_angle(self, servo_number: int) -> int:
        return self.__angles[servo_number]

    def initialize(self, angle=0):
        for i in range(len(self.__servos)):
            self.set_angle(i, angle)

    @classmethod
    def __getMG996R(cls, ServoId: int) -> Servo:
        return Servo(cls.pca.channels[ServoId], **cls.MG996R)

    def __del__(self) -> None:
        self.initialize()
        sleep(0.5)
        for s in self.__servos:
            s.angle = None
