import numpy as np
import cv2


net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "builtin.caffemodel")

image = cv2.imread("4.jpg")
(h, w) = image.shape[:2]

#some magic here. please understand this.
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward() 
#shape of detection is 1,1,num,7 where num is the number detected.
#each of num is a 7-tuple. ignore first two enteries of tuple. 3rd is proability.
#4,5,6,7 are coordiantes in this form startX, startY endX, endY
#notice that, the coordinates are returned as decimals, so they need to be multipled by corresponding
#width and heights to obtain actual coordinates.

for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2] #getting third elem of ith 7-tuple.

	if confidence > 0.5:

		box = detections[0, 0, i, 3:7] #get dimensions as decimals.
		box = box * np.array([w, h, w, h]) #multiples element wise. 
		box = box.astype("int") #casts each element of box np array to int.
		(startX, startY, endX, endY) = box 
		cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10 #to make text stay in place.
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


cv2.imshow("Output", image)
cv2.waitKey(0)