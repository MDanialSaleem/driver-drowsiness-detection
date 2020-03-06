# Basic Procedure
1. Locate faces in videos. This can be done in a single step
with something like builtin detector of OpenCV which uses Single Shot Detector as explained [here](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) . Or this can be done in multiple steps using MTCNN for identification and KCF for tracking. Ideally, this should be done because according to [paper](https://drive.google.com/open?id=1xe6ehO9mmdHH41Fdmgd1D8CARyIXuMxO) this provides the highest accuracy. However, MTCNN is known to be slow.

2. Feed the bounding box of face into a facial landmark detector. A pretrained landmark finder is located in dlib. Otherwise, data for a very detailed landmard detector is [Here](http://www.ifp.illinois.edu/~vuongle2/helen/) . This will prolly need to be trained.

3. After finding landmarks like eye, mouth, we need to feed them into a classifier. A pretrained classifier is given [here](https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/). This project is completely working. However, this uses haar cascade based facial detection which is known to be inefficient. I could not find a pretrained yawn model. The readme of [this project](https://github.com/AnirudhGP/DrowsyDriverDetection) contains links to datasets for eye and yawn. These can be optionally trained or maybe we can find a pretrained model.

The current project contains two files, haar.py which uses a haar based classifier. This classifier can be downloaded from [here](https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/).

Another file is builtin.py which uss the builtin SDD detector found in open cv . This can be found [here](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/).

Day 1,2,5,8,9 of [this course](https://github.com/dloperab/PyImageSearch-CV-DL-CrashCourse) should be read. They are informative.



