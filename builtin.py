import numpy as np
import cv2
import dlib
import sys


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def getMinY(shapeArr, listOfPoints):
    # return the y coordinate of the lowest point in listOfPoints

    minimum = sys.maxsize
    for point in listOfPoints:
        minimum = shapeArr[point][1] if minimum > shapeArr[point][1] else minimum
    return minimum

def getRectCoord(shapeArr, i, j, m, n):
    x = (shapeArr[i][0] + shapeArr[j][0])/2
    x = int(x)
    m = getMinY(shapeArr, m)
    n = getMinY(shapeArr, n)
    y = m + (n - m)/4
    y = int(y)
    return (x,y)

def getEyes(shapeArr):
    lt = getRectCoord(shapeArr, 17,36, [17,21], [37,38])
    lb = getRectCoord(shapeArr, 21, 39, [29], [40,41])
    rt = getRectCoord(shapeArr, 22, 42, [22,26], [43,44])
    rb = getRectCoord(shapeArr, 26,45, [29], [46,47])
    return [(lt,lb),(rt,rb)]



predictor = dlib.shape_predictor("dp.dat")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "builtin.caffemodel")

image = cv2.imread("Images/6.jpg")
(h, w) = image.shape[:2]

# some magic here. please understand this.
blob = cv2.dnn.blobFromImage(cv2.resize(
    image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()
detections = detections[0,0]
# shape of detection is 1,1,num,7 where num is the number detected.
# each of num is a 7-tuple. ignore first two enteries of tuple. 3rd is proability.
# 4,5,6,7 are coordiantes in this form startX, startY endX, endY
# notice that, the coordinates are returned as decimals, so they need to be multipled by corresponding
# width and heights to obtain actual coordinates.

for detection in detections:
    confidence = detection[2]

    if confidence > 0.5:
        box = detection[3:7]
        box = box * np.array([w, h, w, h])
        box = box.astype("int")
        box = [int(x) for x in box] # doing this because the dlib_rectangle function does not accept numpy.int32 types.
        face_location = dlib.rectangle(*box)
        shape = predictor(image, face_location)
        shape = shape_to_np(shape)
        for point in shape:
            cv2.circle(image, tuple(point), 1, (0,0,255), -1)
        
        left, right = getEyes(shape)
        cv2.rectangle(image, left[0], left[1], (0,0,255))
        cv2.rectangle(image, right[0], right[1], (0,0,255))


cv2.imshow("Output", image)
cv2.waitKey(0)
