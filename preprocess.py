
import sys
import numpy as np


def shape_to_np(shape, n):
    # function to convert dlib return to a numpy array.
    coords = np.zeros((n, 2), "int")
    for i in range(0, n):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# for reference to these three functions look at the primary research paper.
def getMinY(shapeArr, listOfPoints):
    # return the y coordinate of the lowest point in listOfPoints
    minimum = sys.maxsize
    for point in listOfPoints:
        minimum = shapeArr[point][1] if minimum > shapeArr[point][1] else minimum
    return minimum


def getMaxY(shapeArr, listOfPoints):
    # return the y coordinate of the highest point in listOfPoints
    maximum = sys.maxsize*-1
    for point in listOfPoints:
        maximum = shapeArr[point][1] if maximum < shapeArr[point][1] else maximum
    return maximum


def getRectCoordPrimary(shapeArr, i, j, m, n):
    x = (shapeArr[i][0] + shapeArr[j][0])/2
    x = int(x)
    m = getMinY(shapeArr, m)
    n = getMinY(shapeArr, n)
    y = m + (n - m)/4
    y = int(y)
    return (x, y)


def getEyesModified(shapeArr):
    leftEyeTopLeft = (shapeArr[17][0], getMinY(shapeArr, [18, 19, 20]))
    leftEyeBottomRight = (shapeArr[21][0], shapeArr[29][1])

    rightEyeTopLeft = (shapeArr[22][0], getMinY(shapeArr, [23, 24, 25]))
    rightEyeBottomRight = (shapeArr[26][0], shapeArr[29][1])

    return [(leftEyeTopLeft, leftEyeBottomRight), (rightEyeTopLeft, rightEyeBottomRight)]


def getEyesPrimary(shapeArr):
    leftEyeTopLeft = getRectCoordPrimary(shapeArr, 17, 36, [17, 21], [37, 38])
    leftEyeBottomRight = getRectCoordPrimary(shapeArr, 21, 39, [29], [40, 41])
    rightEyeTopLeft = getRectCoordPrimary(shapeArr, 22, 42, [22, 26], [43, 44])
    rightEyeBottomRight = getRectCoordPrimary(shapeArr, 26, 45, [29], [46, 47])
    return [(leftEyeTopLeft, leftEyeBottomRight), (rightEyeTopLeft, rightEyeBottomRight)]
