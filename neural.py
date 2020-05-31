from keras.models import load_model
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import cv2
import os
from pathlib import Path

print("[INFO] Loading our own neural network for open close detection")
model = load_model("Models/nonet.hdf5")
print("[INFO] Complete")



class Preprocessor:
    def __init__(self, width, height, interpolAlgo=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.intterpolAlgo = interpolAlgo

    def process(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.intterpolAlgo)


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def process(self, image):
        return img_to_array(image, data_format=self.dataFormat)




def predict(data):
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    sp = Preprocessor(24, 24)
    iap = ImageToArrayPreprocessor()
    data = sp.process(data)
    data = iap.process(data)
    data = data.reshape(1, 24, 24, 3)
    preditions = model.predict(data)
    return preditions
