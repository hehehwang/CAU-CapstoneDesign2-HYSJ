from typing import Iterable, List, Union

import cv2
import mediapipe as mp
import numpy as np
from google.protobuf.pyext._message import RepeatedCompositeContainer
from mediapipe.framework.formats.landmark_pb2 import Landmark
from numpy import ndarray

from .poseLandmarks import LANDMARK_ID_TO_NAME, LANDMARK_NAME_TO_ID


class MP_Pose:
    MEDIAPIPE_POSE = mp.solutions.pose

    def __init__(self):
        self.pose = MP_Pose.MEDIAPIPE_POSE.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.__cap = cv2.VideoCapture(0)
        self.__poseGen = self.__getPoseGenerator()

    def __getFrame(self) -> ndarray:
        while 1:
            ret, frame = self.__cap.read()
            if ret:
                break

        return frame

    def __getPoseGenerator(self) -> Iterable[type]:
        while 1:
            frame = self.__getFrame()
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            yield results

    def getPoseWorldLandmarks(self) -> RepeatedCompositeContainer:
        results = next(self.__poseGen)
        pwlm = results.pose_world_landmarks.landmark

        return pwlm

    def getLandmarks(self, ids_or_names: List[Union[str, int]]) -> List[Landmark]:
        if isinstance(ids_or_names[0], str):
            ids = [LANDMARK_NAME_TO_ID[s] for s in ids_or_names]
        else:
            ids = ids_or_names
        landmarks = self.getPoseWorldLandmarks()
        result = [landmarks[i] for i in ids]

        return result
