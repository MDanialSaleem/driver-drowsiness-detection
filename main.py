import numpy as np
import cv2
from keras.models import load_model

import dlib
import sys
import time
import argparse

from preprocess import getEyes, shape_to_np
from neural import predict as PredictOpennes

print("[INFO] Loading our own neural network for open close detection")
model = load_model("Models/dddshallow.hdf5")
print("[INFO] Complete")

parser = argparse.ArgumentParser(description='Driver Drowsiness Detection')
parser.add_argument('--impath', default="video", type=str)
impath = parser.parse_args().impath


print("[INFO] Loading DLIB shape predictor 5 and 68")
DLIB5 = dlib.shape_predictor("Models/DLIB5.dat")
DLIB68 = dlib.shape_predictor("Models/DLIB68.dat")
print("[INFO] Complete")
print("[INFO] Loading SSD")
net = cv2.dnn.readNetFromCaffe(
    "Models/deploy.prototxt.txt", "Models/builtin.caffemodel")
print("[INFO] Complete")

if impath == "video":
    print("[INFO] Loading camera")
    cam = cv2.VideoCapture(0)
    time.sleep(2)
    print("[INFO] complete")

count = 0

while True:
    if impath == "video":
        retval, image = cam.read()
    else:
        image = cv2.imread(impath)
    (h, w) = image.shape[:2]

    # some magic here. please understand this.
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    detections = detections[0, 0]
    # shape of detection is 1,1,num,7 where num is the number detected.
    # each of num is a 7-tuple. ignore first two enteries of tuple. 3rd is proability.
    # 4,5,6,7 are coordiantes in this form startX, startY endX, endY
    # notice that, the coordinates are returned as decimals, so they need to be multipled by corresponding
    # width and heights to obtain actual coordinates.

    for detection in detections:
        confidence = detection[2]

        if confidence > 0.2:
            box = detection[3:7]
            box = box * np.array([w, h, w, h])
            box = box.astype("int")
            padder = 5
            box += padder
            # doing this because the dlib_rectangle function does not accept numpy.int32 types.
            box = [int(x) for x in box]
            face_location = dlib.rectangle(*box)
            shape5 = DLIB5(image, face_location)
            shape5 = shape_to_np(shape5, 5)
            shape68 = DLIB68(image, face_location)
            shape68 = shape_to_np(shape68, 68)

            left68, right68 = getEyes(shape68)

            x1, y1 = left68[0]
            x2, y2 = left68[1]
            paddingTop = 20
            paddingBottom = 20
            x1 -= paddingTop
            y1 -= paddingTop
            x2 += paddingBottom
            y2 += paddingBottom
            e = image[y1:y2, x1:x2]
            res = PredictOpennes(e, model)
            if res is None:
                continue

            res = res[0][0]
            if res != 0:
                count += 1
            
            cv2.putText(image, str(count), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255))

            cv2.rectangle(image, left68[0], left68[1], (0, 0, 255))
            cv2.rectangle(image, right68[0], right68[1], (0, 0, 255))
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
