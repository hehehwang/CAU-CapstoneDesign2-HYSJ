import logging
from time import sleep
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from .landmarkHandler import MyLandmark, Vector3d
from .mediapipeHandler import MP_Holistic


class RecognizedArm(NamedTuple):
    image: np.ndarray
    pose: Dict[str, MyLandmark]
    hand: Dict[str, MyLandmark]
    d: List[float]
    theta: List[float]


class MyArm:
    def __init__(self, isRightArm=True, visThreshold=0.80, imgDir=None) -> None:
        self.__armSide = "right" if isRightArm else "left"
        self.__visThreshold = visThreshold
        self.__mp = MP_Holistic(imgDir)

    def __getMPRs(self):
        return self.__mp.getMPResult()

    def process(self):

        while 1:
            mpr = self.__getMPRs()
            img, landmarks = mpr.image, mpr.results
            if not landmarks.pose[self.__armSide]:
                logging.warning("No pose detected")
                sleep(0.5)
                continue
            if not landmarks.hand[self.__armSide]:
                logging.warning("No hand detected")
                sleep(0.5)
                continue
            pose = {
                i: landmarks.pose[self.__armSide][i]
                for i in ["shoulder", "elbow", "wrist", "pinky", "index"]
            }
            hand = {
                i: landmarks.hand[self.__armSide][i]
                for i in ["INDEX_FINGER_TIP", "THUMB_TIP"]
            }

            isVisable = all([p.vis >= self.__visThreshold for p in pose.values()])
            if not isVisable:
                logging.warning("Not all landmarks are visible")
                w = ""
                for p in pose:
                    if pose[p].vis < self.__visThreshold:
                        w += f"{p} is not visible, {pose[p].vis} \n"
                logging.warning(w)
                sleep(0.5)
                continue

            # print(f'{pose["wrist"]=}')
            origin = pose["elbow"].np
            for p in pose:
                v = pose[p]
                pose[p] = Vector3d((v.np - origin))
            handTip = Vector3d((pose["index"].np + pose["pinky"].np) / 2)

            elbowToWrist_projected = Vector3d(
                pose["wrist"].np - np.array([0, 0, pose["wrist"].z])
            )

            theta0 = Vector3d([1, 0, 0]).angleBtw(elbowToWrist_projected)
            theta1 = pose["wrist"].angleBtw(Vector3d(0, -1, 0))
            wristToHandTip = Vector3d(handTip.np - pose["wrist"].np)
            # print(f'{pose["wrist"]=}')
            # print(f"{wristToHandTip=}")
            theta2 = pose["wrist"].angleBtw(wristToHandTip)
            d = hand["THUMB_TIP"].distBtw(hand["INDEX_FINGER_TIP"])
            return RecognizedArm(img, pose, hand, [d], [theta0, theta1, theta2])
