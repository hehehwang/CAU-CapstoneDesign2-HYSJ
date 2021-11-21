from typing import List

import pandas as pd
from mediapipe.framework.formats.landmark_pb2 import Landmark

from .mediapipeHandler import MP_Pose


class RecogArm:
    RIGHT_ARM = [14, 16, 18, 20, 22]
    LEFT_ARM = [13, 15, 17, 19, 21]
    LABEL = ["elbow", "wrist", "pinky", "index", "thumb"]

    def __init__(self, isRightArm: bool = True, threshold: float = 0.8):
        self.MPP = MP_Pose()
        self.myArm = RecogArm.RIGHT_ARM if isRightArm else RecogArm.LEFT_ARM
        self.threshold = threshold

    def __collectMyArmLandmarks(self, landmarks: List[Landmark]) -> List[Landmark]:
        MyArmLandmarks = [landmarks[i] for i in self.myArm]
        return MyArmLandmarks

    def isAllLandmarksVisible(self, landmarks: List[Landmark]) -> bool:
        return all([l.visibility > self.threshold for l in landmarks])

    def getMyArmsImangeAndDF(self):
        image, landmarks = self.MPP.getImageAndWorldLandmarks()
        landmarks = self.__collectMyArmLandmarks(landmarks)

        data = {"x": [], "y": [], "z": [], "label": [], "vis": []}
        elbox, elboy, elboz = landmarks[0].x, landmarks[0].y, landmarks[0].z
        for i, l in enumerate(landmarks):
            data["x"].append(l.x - elbox)
            data["y"].append(l.y - elboy)
            data["z"].append(l.z - elboz)
            data["label"].append(RecogArm.LABEL[i])
            data["vis"].append(round(l.visibility, 2))
        df = pd.DataFrame(data)
        return image, df
