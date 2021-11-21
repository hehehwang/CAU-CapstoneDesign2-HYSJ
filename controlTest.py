from time import sleep

from board import SCL, SDA
import busio

from adafruit_motor.servo import Servo
from adafruit_pca9685 import PCA9685

i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50
MG996R = {'min_pulse':700, 'max_pulse':2400}
def main():
    angleTest()

def angleTest():
    myServo = Servo(pca.channels[0], **MG996R)
    print("zero degree")
    myServo.angle = 0
    sleep(3)
    # print("30 degree")
    # myServo.angle = 30
    # sleep(3)
    # print("45 degree")
    # myServo.angle = 45
    # sleep(3)
    # print("60 degree")
    # myServo.angle = 60
    # sleep(3)
    print("90 degree")
    myServo.angle = 90 
    sleep(3)
    print("180 degree")
    myServo.angle = 180
    sleep(3)
    print("Done")
    myServo.angle = 0
    sleep(3)
    myServo.angle = None

def servoTest():
    myServo = Servo(pca.channels[2], **MG996R)
    for i in range(0, 180, 1):
        myServo.angle = i
        print(i)
        sleep(0.01)
    sleep(0.5)
    for i in range(180, 0, -1):
        myServo.angle = i
        print(i)
        sleep(0.01)
    sleep(0.5)
    myServo.angle = None

def armTest():
    sleep(3)
    baseServo = Servo(pca.channels[0], **MG996R)
    sleep(0.5)
    handServo = Servo(pca.channels[1], **MG996R)
    print(pca.channels[0], pca.channels[1], handServo)

    for i in range(40, 90, 1):
        baseServo.angle = i
        print(i)
        sleep(0.01)
    for i in range(90, 30, -1):
        baseServo.angle = i
        print(i)
        sleep(0.01)

    handServo.angle = 10
    sleep(0.5)
    handServo.angle = 30
    sleep(3)
    baseServo.angle = None
    sleep(1)
    handServo.angle = None

if __name__ == "__main__":
    main()
    # armTest()