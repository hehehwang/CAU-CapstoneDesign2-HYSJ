from typing import Any, Iterator, List, NamedTuple, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import Landmark
from numpy import ndarray

from .CONST import POSE_LANDMARK_ID_TO_NAME, POSE_LANDMARK_NAME_TO_ID
from .landmarkHandler import HolisticLandmarks

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


class MediapipeResult(NamedTuple):
    image: np.ndarray
    rawImage: np.ndarray
    results: HolisticLandmarks
    rawResults: type


class Webcam:
    def __init__(self) -> None:
        from time import perf_counter

        # pt = perf_counter()
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # npt = perf_counter()
        # print(f"Webcam init took {npt - pt} seconds")
        # pt = npt
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        # npt = perf_counter()
        # print(f"Webcam set fourcc took {npt - pt} seconds")
        # pt = npt
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # npt = perf_counter()
        # print(f"Webcam set resolution took {npt - pt} seconds")
        # pt = npt
        capture.set(cv2.CAP_PROP_FPS, 60)
        # npt = perf_counter()
        # print(f"Webcam set fps took {npt - pt} seconds")

        self.__cap = capture

    @property
    def frame(self) -> ndarray:
        while 1:
            ret, frame = self.__cap.read()
            if ret:
                break
        frame = cv2.flip(frame, 1)

        return frame

    def __del__(self) -> None:
        self.__cap.release()


class MP_Holistic:
    MEDIAPIPE_HOLISTIC = mp.solutions.holistic
    MP_DRAWING = mp.solutions.drawing_utils
    MP_DRAWING_STYLES = mp.solutions.drawing_styles

    def __init__(
        self,
        imageDirectory=None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # self.isImage, self.__camera, self.__inputImg = False, None, None
        # if not imageDirectory:
        #     self.__camera = Webcam()
        # else:
        #     self.__inputImg = cv2.imread(imageDirectory, cv2.IMREAD_COLOR)
        #     self.isImage = True

        self.__nextFrame = self.__getFrameSource(imageDirectory)
        self.__holistic = MP_Holistic.MEDIAPIPE_HOLISTIC.Holistic(
            # static_image_mode=self.isImage,
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __getFrameSource(self, imageDirectory: Optional[str]) -> Iterator[ndarray]:
        __camera = None
        __img = None
        while 1:
            if imageDirectory:
                if __img is None:
                    __img = cv2.imread(imageDirectory, cv2.IMREAD_COLOR)
                yield __img
            else:
                if __camera is None:
                    __camera = Webcam()
                yield __camera.frame

    def getMPResult(self) -> MediapipeResult:
        bgrImage = next(self.__nextFrame)

        rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
        mpResults = self.__holistic.process(rgbImage)
        hlms = HolisticLandmarks(mpResults)
        processedBgrImage = self.__showResultsOnFrame(bgrImage.copy(), mpResults)
        return MediapipeResult(
            image=processedBgrImage,
            rawImage=bgrImage,
            results=hlms,
            rawResults=mpResults,
        )

    def __showResultsOnFrame(self, bgrImage: ndarray, results) -> ndarray:
        """
        Draws the landmarks on the frame.
        """

        MP_Holistic.MP_DRAWING.draw_landmarks(
            bgrImage,
            results.pose_landmarks,
            MP_Holistic.MEDIAPIPE_HOLISTIC.POSE_CONNECTIONS,
            landmark_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_pose_landmarks_style(),
        )
        MP_Holistic.MP_DRAWING.draw_landmarks(
            bgrImage,
            results.left_hand_landmarks,
            MP_Holistic.MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS,
            landmark_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
            connection_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_hand_connections_style(),
        )
        MP_Holistic.MP_DRAWING.draw_landmarks(
            bgrImage,
            results.right_hand_landmarks,
            MP_Holistic.MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS,
            landmark_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
            connection_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_hand_connections_style(),
        )
        return bgrImage

    def showResult(self) -> None:
        while 1:
            r = self.getMPResult()
            cv2.imshow("MediaPipe Holistic", r.image)
            if cv2.waitKey(2) & 0xFF == 27:
                break
