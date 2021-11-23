from pprint import pprint

from components.dataclass import HolisticLandmarks
from components.mediapipeHandler import MP_Holistic

mph = MP_Holistic("./sampleImg/sample1.jpg")
# mph = MP_Holistic()
# mph.showResult()
img, result = mph.getMPResult()
while not (result.left_hand_landmarks or result.right_hand_landmarks):
    img, result = mph.getMPResult()
    print("retry..")
# print(dir(result))
# print(result.face_landmarks)
hlm = HolisticLandmarks(result)
pprint(hlm.landmarks)
