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
    image = cv2.putText(
        r.image,
        str(round(r.d[0], 2)),
        (800, 200),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 0, 0),
        2,
    )
    image = cv2.putText(
        image,
        ", ".join(map(lambda x: str(round(x, 2)), r.theta)),
        (800, 100),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 0, 0),
        2,
    )
    cv2.imshow("MediaPipe Holistic", image)
    # pprint(r.d)
    # pprint(r.theta)
    # print("=" * 20)
    if cv2.waitKey(100) & 0xFF == 27:
        break
