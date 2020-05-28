from neural import predict as PredictOpennes
from facedetection import getFace
from eyes_detection import getEyesBB, extractROI
import numpy as np
import cv2
import sys
import time
import argparse
from pathlib import Path
import os


parser = argparse.ArgumentParser(
    description='Driver Drowsiness Detection Test')
parser.add_argument('--imfolder', default="./Images/", type=str)
imfolder = parser.parse_args().imfolder
folder = Path(imfolder)

count2 = 0
count1 = 0
count0 = 0
lCount = 0
rCount = 0
for thing in folder.iterdir():
    if thing.is_file():
        image = cv2.imread(str(thing))
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        for box in getFace(blob, w, h):
            left, right = getEyesBB(image, box)
            lROI, rROI = extractROI(image, left, right)

            lResult = PredictOpennes(lROI)[0][0]
            rResult = PredictOpennes(rROI)[0][0]

            if lResult:
                lCount += 1

            if rResult:
                rCount += 1

            if lResult and rResult:
                count2 += 1
            elif lResult or rResult:
                count1 += 1
            else:
                count0 += 1

print("Count0: " + str(count0))
print("Count1: " + str(count1))
print("Count2: " + str(count2))
print("Left: " + str(lCount))
print("Right: " + str(rCount))
