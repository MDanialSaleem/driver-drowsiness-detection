from neural import predict as PredictOpennes
from facedetection import getFace
from eyes_detection import getEyesBB, extractROI
import numpy as np
import cv2
import winsound

import sys
import time

print("[INFO] Loading camera")
cam = cv2.VideoCapture(0)
time.sleep(2)
print("[INFO] complete")

lCount = 0
rCount = 0

closedCount = 0
alarmRunning = False

while True:
    retval, image = cam.read()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    for box in getFace(blob, w, h):
        left, right = getEyesBB(image, box)
        lROI, rROI = extractROI(image, left, right)

        lResult = PredictOpennes(lROI)[0][0]
        rResult = PredictOpennes(rROI)[0][0]

        lCount = lCount + 1 if lResult != 0 else lCount
        rCount = rCount + 1 if rResult != 0 else rCount
        threshold = 0.01
        if lResult > threshold or rResult > threshold:
            closedCount = 0
            if alarmRunning:
                alarmRunning = False
                winsound.PlaySound(None, winsound.SND_ASYNC)

        if lResult <= threshold and rResult <= threshold and not alarmRunning:
            closedCount += 1

        if closedCount > 10 and not alarmRunning:
            alarmRunning = True
            winsound.PlaySound(
                "Alarm.wav", winsound.SND_ASYNC | winsound.SND_FILENAME | winsound.SND_ALIAS)

        cv2.rectangle(image, left[0], left[1], (0, 0, 255))
        cv2.rectangle(image, right[0], right[1], (0, 0, 255))
        cv2.rectangle(image, (box[0], box[1]),
                      (box[2], box[3]), (255, 0, 0,))

    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
