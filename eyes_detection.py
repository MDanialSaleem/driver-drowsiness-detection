import dlib
from preprocess import getEyes, shape_to_np


print("[INFO] Loading DLIB shape predictor 5 and 68")
# DLIB5 = dlib.shape_predictor("Models/DLIB5.dat")
DLIB68 = dlib.shape_predictor("Models/DLIB68.dat")
print("[INFO] Complete")


# this is for dlib5 if we ever decide to use that.
# shape5 = DLIB5(image, face_location)
# shape5 = shape_to_np(shape5, 5)

def getEyesBB(image, box):
    face_location = dlib.rectangle(*box)
    shape68 = DLIB68(image, face_location)
    shape68 = shape_to_np(shape68, 68)
    left68, right68 = getEyes(shape68)
    return left68, right68


def extractROI(image, left, right):

    def getEach(image, eye):
        x1, y1 = eye[0]
        x2, y2 = eye[1]
        paddingTop = 20
        paddingBottom = 20
        x1 -= paddingTop
        y1 -= paddingTop
        x2 += paddingBottom
        y2 += paddingBottom
        ROI = image[y1:y2, x1:x2]
        return ROI

    return getEach(image, left), getEach(image, right)
