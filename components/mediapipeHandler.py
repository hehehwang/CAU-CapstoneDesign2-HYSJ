from typing import Any, Iterator, List, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import Landmark
from numpy import ndarray

from .poseLandmarks import LANDMARK_ID_TO_NAME, LANDMARK_NAME_TO_ID

"""
results:
<class 'type'>
['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__module__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmul__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '_asdict', '_field_defaults', '_fields', '_fields_defaults', '_make', '_replace', 'count', 'index', 'pose_landmarks', 'pose_world_landmarks', 'segmentation_mask']

results.pose_wrorld_landmarks:
<class 'mediapipe.framework.formats.landmark_pb2.LandmarkList'> ['ByteSize', 'Clear', 'ClearExtension', 'ClearField', 'CopyFrom', 'DESCRIPTOR', 'DiscardUnknownFields', 'Extensions', 'FindInitializationErrors', 'FromString', 'HasExtension', 'HasField', 'IsInitialized', 'ListFields', 'MergeFrom', 'MergeFromString', 'ParseFromString', 'RegisterExtension', 'SerializePartialToString', 'SerializeToString', 'SetInParent', 'UnknownFields', 'WhichOneof', '_CheckCalledFromGeneratedFile', '_SetListener', '__class__', '__deepcopy__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__unicode__', '_extensions_by_name', '_extensions_by_number', 'landmark']

results.pose_world_landmarks.landmark:
<class 'google.protobuf.pyext._message.RepeatedCompositeContainer'>
['MergeFrom', '__class__', '__deepcopy__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'add', 'append', 'extend', 'insert', 'pop', 'remove', 'reverse', 'sort']

results.pose_world_landmarks.landmark[0]:
<class 'mediapipe.framework.formats.landmark_pb2.Landmark'>
['ByteSize', 'Clear', 'ClearExtension', 'ClearField', 'CopyFrom', 'DESCRIPTOR', 'DiscardUnknownFields', 'Extensions', 'FindInitializationErrors', 'FromString', 'HasExtension', 'HasField', 'IsInitialized', 'ListFields', 'MergeFrom', 'MergeFromString', 'ParseFromString', 'RegisterExtension', 'SerializePartialToString', 'SerializeToString', 'SetInParent', 'UnknownFields', 'WhichOneof', '_CheckCalledFromGeneratedFile', '_SetListener', '__class__', '__deepcopy__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__unicode__', '_extensions_by_name', '_extensions_by_number', 'presence', 'visibility', 'x', 'y', 'z']
"""


class MP_Pose:
    MEDIAPIPE_POSE = mp.solutions.pose
    MP_DRAWING = mp.solutions.drawing_utils
    MP_DRAWING_STYLES = mp.solutions.drawing_styles

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.pose = MP_Pose.MEDIAPIPE_POSE.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        capture = cv2.VideoCapture(0)
        # capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        self.__cap = capture

        self.__poseGen = self.__getPoseGenerator()

    def __getFrame(self) -> ndarray:
        while 1:
            ret, frame = self.__cap.read()
            if ret:
                break
        frame = cv2.flip(frame, 1)

        return frame

    def __getPoseGenerator(self) -> Iterator[mp.Packet]:
        while 1:
            frame = self.__getFrame()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            yield results

    def __getMpResultFromFrame(self, frame: ndarray) -> mp.Packet:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        return results

    def __getMarkedImageFromFrame(self, frame: ndarray, results: mp.Packet) -> ndarray:
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        MP_Pose.MP_DRAWING.draw_landmarks(
            image,
            results.pose_landmarks,
            MP_Pose.MEDIAPIPE_POSE.POSE_CONNECTIONS,
            landmark_drawing_spec=MP_Pose.MP_DRAWING_STYLES.get_default_pose_landmarks_style(),
        )

        return image

    def getImageAndWorldLandmarks(self) -> Tuple[ndarray, List[Landmark]]:
        frame = self.__getFrame()
        results = self.__getMpResultFromFrame(frame)
        landmarks = [l for l in results.pose_world_landmarks.landmark]
        processedImage = self.__getMarkedImageFromFrame(frame, results)

        return processedImage, landmarks
