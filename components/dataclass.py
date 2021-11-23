from typing import Union

import numpy as np
from mediapipe.framework.formats.landmark_pb2 import Landmark
from numpy.core.fromnumeric import prod

from .CONST import (HAND_LANDMARK_ID_TO_NAME, HAND_LANDMARK_NAME_TO_ID,
                    POSE_LANDMARK_ID_TO_NAME, POSE_LANDMARK_NAME_TO_ID)


class Vector3d:
    def __init__(self, *args: Union[int, float, np.ndarray]):
        if len(args) == 1:
            self.__np = np.array(args[0], dtype=np.float64)
        elif len(args) == 2:
            self.__np = np.array([args[0], args[1], 0], dtype=np.float64)
        elif len(args) == 3:
            self.__np = np.array(args, dtype=np.float64)
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
        self.__x = self.__np[0]
        self.__y = self.__np[1]
        self.__z = self.__np[2]

    def unitVector(self) -> "Vector3d":
        ans = self.asNp / np.linalg.norm(self.asNp)
        return Vector3d(ans[0], ans[1], ans[2])

    def angleBtw(self, other: "Vector3d") -> float:
        unitVectorA = self.unitVector()
        unitVectorB = other.unitVector()
        product = np.dot(unitVectorA.asNp, unitVectorB.asNp)
        if product >= 1:
            return 0.0
        if product <= -1:
            return 180.0
        angleRad = np.arccos(product)
        return np.rad2deg(angleRad)

    def distBtw(self, other: "Vector3d") -> float:
        return np.linalg.norm(self.asNp - other.asNp)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Vector3d):
            return np.isclose(self.asNp, __o.asNp).all()
        return False

    @property
    def x(self) -> np.float64:
        return self.__x

    @property
    def y(self) -> np.float64:
        return self.__y

    @property
    def z(self) -> np.float64:
        return self.__z

    @property
    def asNp(self) -> np.ndarray:
        return self.__np

    def __str__(self) -> str:
        return f"({self.__x}, {self.__y}, {self.__z})"

    def __repr__(self) -> str:
        return self.__str__()


class MyLandmark(Vector3d):
    def __init__(self, *args):
        if len(args) == 4:
            __x, __y, __z, __vis = args
            super().__init__(__x, __y, __z)
        elif len(args) == 2:
            __x, __y = args
            __vis = 1.0
            super().__init__(__x, __y, 0)
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
        self.__vis = __vis

    @property
    def vis(self) -> Union[int, float]:
        return self.__vis

    def __str__(self) -> str:
        return f"{super().__str__()}: {self.__vis}"

    def __repr__(self) -> str:
        return self.__str__()


class HolisticLandmarks:
    def __init__(self, MediapipeResult):
        (
            __poseWorldLandmarks,
            __lefthandLandmarks,
            __righthandLandmarks,
        ) = HolisticLandmarks.__preprocessLandmarks(MediapipeResult)
        self.__landmarks = {
            "leftSide": {"pose": {}, "hand": {}},
            "rightSide": {"pose": {}, "hand": {}},
        }
        for i, pwlm in enumerate(__poseWorldLandmarks):
            landmark = MyLandmark(pwlm.x, pwlm.y, pwlm.z, pwlm.visibility)
            dest = self.__landmarks["leftSide" if i % 2 else "rightSide"]
            dest["pose"][POSE_LANDMARK_ID_TO_NAME[i]] = landmark

        for i, hlm in enumerate(__lefthandLandmarks):
            landmark = MyLandmark(hlm.x, hlm.y)
            dest = self.__landmarks["leftSide"]["hand"]
            dest[HAND_LANDMARK_ID_TO_NAME[i]] = landmark

        for i, hlm in enumerate(__righthandLandmarks):
            landmark = MyLandmark(hlm.x, hlm.y)
            dest = self.__landmarks["rightSide"]["hand"]
            dest[HAND_LANDMARK_ID_TO_NAME[i]] = landmark

    @staticmethod
    def __preprocessLandmarks(MediapipeResults) -> tuple:
        poseWorldLandmarks, lefthandLandmarks, righthandLandmarks = [], [], []
        if MediapipeResults.pose_world_landmarks:
            poseWorldLandmarks = MediapipeResults.pose_world_landmarks.landmark
        if MediapipeResults.left_hand_landmarks:
            lefthandLandmarks = MediapipeResults.left_hand_landmarks.landmark
        if MediapipeResults.right_hand_landmarks:
            righthandLandmarks = MediapipeResults.right_hand_landmarks.landmark
        return poseWorldLandmarks, lefthandLandmarks, righthandLandmarks

    @property
    def landmarks(self) -> dict:
        return self.__landmarks

    def __str__(self) -> str:
        from pprint import pformat

        return pformat(self.__landmarks)

    def __repr__(self) -> str:
        return self.__str__()
