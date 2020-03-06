import cv2
from keras.models import load_model
import time


face = cv2.CascadeClassifier('cascade.xml')


model = load_model('models/cnncat2.h5')

frame = cv2.imread("2.jpg")

height,width = frame.shape[:2] 

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))

cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

for (x,y,w,h) in faces:
     cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

cv2.imshow('frame',frame)
cv2.waitKey(0)