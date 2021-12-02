import json
from pprint import pprint

import cv2

# from components.landmarkHandler import HolisticLandmarks, Vector3d
# from components.mediapipeHandler import MP_Holistic
from components.myArm import MyArm
from components.Server import Server

ma = MyArm(isRightArm=True, visThreshold=0.70)
server = Server()


def distanceToAngle(distance):
    # 0.26 -> 0
    # 0.01 -> 50
    max_distance, min_distance = 0.24, 0.01
    distance = max(distance, min_distance)
    distance = min(distance, max_distance)
    angle = (1 - ((distance - min_distance) / (max_distance - min_distance))) * 50
    return angle


while 1:
    msg = server.receive()
    print(msg)
    r = ma.process()
    image = cv2.putText(
        r.image,
        str(round(r.d[0], 2)),
        (800, 200),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 0, 0),
        2,
    )
    # image = cv2.putText(
    #     image,
    #     ", ".join(map(lambda x: str(round(x, 2)), r.theta)),
    #     (800, 100),
    #     cv2.FONT_HERSHEY_PLAIN,
    #     3,
    #     (0, 0, 0),
    #     2,
    # )
    cv2.imshow("MediaPipe Holistic", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    angle0 = distanceToAngle(r.d[0])
    print(f"{r.d[0]}, {angle0}")
    server.send(json.dumps([angle0]))
    # print(msg)
    # angle = int(input())
    # if angle == -1:
    #     server.send("exit")
    #     break
    # server.send(json.dumps([angle]))
