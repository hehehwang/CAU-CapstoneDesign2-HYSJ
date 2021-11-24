from pprint import pprint

import cv2

from components.landmarkHandler import HolisticLandmarks, Vector3d
from components.mediapipeHandler import MP_Holistic
from components.myArm import MyArm

# mph = MP_Holistic("./sampleImg/sample1.jpg")
# mph = MP_Holistic()
# mph.showResult()
# mpr = mph.getMPResult()
# img, result = mpr.image, mpr.results
# while not (result.hand["left"] or result.hand["right"]):
#     img, result = mph.getMPResult()
#     print("retry..")
# pprint(result)

ma = MyArm(isRightArm=True, visThreshold=0.70)
while 1:
    r = ma.process()
    cv2.imshow("MediaPipe Holistic", r.image)
    # pprint(r.d)
    pprint(r.theta)
    print("=" * 20)
    if cv2.waitKey(500) & 0xFF == 27:
        break
