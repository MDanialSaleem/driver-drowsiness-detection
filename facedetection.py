import cv2
import numpy as np
print("[INFO] Loading SSD")
net = cv2.dnn.readNetFromCaffe(
    "Models/deploy.prototxt.txt", "Models/builtin.caffemodel")
print("[INFO] Complete")


def getFace(blob, w, h):
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
                yield box
