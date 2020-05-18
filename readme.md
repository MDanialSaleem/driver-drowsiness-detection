
# Quick Setup 

## Getting models
Download [this](https://drive.google.com/open?id=1QCXmKlSQidpg35FxxOdB4Cz_tvecUA9s) folder. And store all the files in Models directory within your project.

## Creating env.
```
conda env create -f environment.yml
```

## Runnng code
Activate the env.

```
conda activate ddd
```

To run on webcam

```
python main.py
```

To run with image

```
python main.py --impath="pathtoimage"
```

Left and right shows the number of time the detector is counter the respective eye to be open.


# Basic Procedure
1. Locate faces in videos. This can be done in a single step
with something like builtin detector of OpenCV which uses Single Shot Detector as explained [here](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) . This has
been implemented but this causes issues with detection for some side faces.

2. Feed the bounding box of face into a facial landmark detector. A pretrained landmark finder is located in dlib. This has been implemented.

3. After finding landmarks like eye, mouth, we need to feed them into a classifier. A pretrained classifier is given [here](https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/). This project is completely working. However, this uses haar cascade based facial detection which is known to be inefficient. I could not find a pretrained yawn model. Dataset was obtained from [here](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html). We used the one with only eye patches. This dataset was trained on primitive model. Currently, accuracy is nearly 95% with no signs of overfitting. Our own trained model can be downloaded from [here] (https://drive.google.com/open?id=1byaquY91zGXs00P6UAIgTg3yQzQCO9zk)

The current project contains two files, haar.py which uses a haar based classifier. This classifier can be downloaded from [here](https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/).

Another file is main.py which uss the builtin SDD detector found in open cv and the two facial landmark detector found in dlib. SSD can be found [here](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/). Dlib models can be found [here](https://github.com/davisking/dlib-models). We use the facial landmark predictor 68 and 5. 5 will prolly not be used. 
Just using it now it now to see if we can get maybe better accuracy with it.

As it stands, all parts are completed. But they require more accuracy. Part 2 has been done exactly
like our primary paper. Part one has been chnaged from MTCNN+KCF to SSD. 
As far as the issue with accuracy is concerned, we can try tackling that by using face detectors 
built into dlib (there is HOG+SVN based detector and a CNN based detector). Creator of dlid mentions
that the landmark detector performs better if we use the dlib detectors themselves. Or try anything
else but we need to improve some accuracy at least. First of all try increase the accuacy of part 1
and then well see about part 2.

To run the file on web cam, run it without any arguments. Otherwise, use --impath=pathtoimage to run this on an image

Day 1,2,5,8,9 of [this course](https://github.com/dloperab/PyImageSearch-CV-DL-CrashCourse) should be read. They are informative.



