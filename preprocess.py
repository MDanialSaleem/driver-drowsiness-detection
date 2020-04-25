
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