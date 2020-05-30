from neural import predict as PredictOpennes
from facedetection import getFace
from eyes_detection import getEyesBB, extractROI
import numpy as np
import cv2
import winsound

import sys
import time
import argparse

parser = argparse.ArgumentParser(description='Driver Drowsiness Detection')
parser.add_argument('--impath', default="video", type=str)
impath = parser.parse_args().impath


if impath == "video":
    print("[INFO] Loading camera")
    cam = cv2.VideoCapture(0)
    time.sleep(2)
    print("[INFO] complete")

lCount = 0
rCount = 0

closedCount = 0
alarmRunning = False

while True:
    if impath == "video":
        retval, image = cam.read()
    else:
        image = cv2.imread(impath)
    (h, w) = image.shape[:2]

    # some magic here. please understand this.
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    for box in getFace(blob, w, h):
        left, right = getEyesBB(image, box)
        lROI, rROI = extractROI(image, left, right)

        lResult = PredictOpennes(lROI)[0][0]
        rResult = PredictOpennes(rROI)[0][0]

        lCount = lCount + 1 if lResult != 0 else lCount
        rCount = rCount + 1 if rResult != 0 else rCount

        if lResult > 0 or rResult > 0:
            closedCount = 0
            if alarmRunning:
                alarmRunning = False
                winsound.PlaySound(None, winsound.SND_ASYNC)

        if lResult == 0 and rResult == 0 and not alarmRunning:
            closedCount += 1

        if closedCount > 10:
            if not alarmRunning:
                alarmRunning = True
                winsound.PlaySound(
                    "Alarm.wav", winsound.SND_ASYNC | winsound.SND_FILENAME | winsound.SND_ALIAS)
            cv2.putText(image, "Sleepy", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

        cv2.rectangle(image, left[0], left[1], (0, 0, 255))
        cv2.rectangle(image, right[0], right[1], (0, 0, 255))
        cv2.rectangle(image, (box[0], box[1]),
                      (box[2], box[3]), (255, 0, 0,))

    cv2.imshow("Output", image)
    if not impath == "video":
        cv2.waitKey()
        break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if impath == 'video':
    cam.release()
cv2.destroyAllWindows()
