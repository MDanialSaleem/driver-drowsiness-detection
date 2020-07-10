A deep learning model built with Keras that checks for drowsiness in a driver and rings an alarm if it crosses a (configureable) threshold. As of now, it only takes into account the state of the eyes. However, using the state of the mouth
is also a viable option.

## Quick Setup

- Download [this](https://drive.google.com/open?id=1QCXmKlSQidpg35FxxOdB4Cz_tvecUA9s) folder. And store all the files in Models directory within your project.

- Creating the environment

```
conda env create -f environment.yml
conda activate ddd
```

- Running the code

```
python main.py
```

## Procedure

1. Locate faces in videos. This can be done in a single step
   with something like builtin detector of OpenCV which uses Single Shot Detector. Alternatives include using YOLO, Faster RCNN etc. However, SSD is found to have a nice balance of accuracy and speed.

2. Feed the bounding box of face into a facial landmark detector. A pretrained landmark finder is located in dlib. There are two options regarding this, the 68 point complete detector or the 5 point smaller detector. The 5-point cannot be used to find bounding boxes of eyes so 68 is used is here.

3. After finding landmarks like eye, mouth, we need to feed them into a classifier. Dataset was obtained from [here](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html). We used the one with only eye patches.


### Dataset
Dataset was obtained from [here](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html). We used the one with only eye patches. The lack of data, and the lack of variety in it caused some problems in wild detection.

### Model

[![model-summary.png](https://i.postimg.cc/TYcRG1ZZ/model-summary.png)](https://postimg.cc/nCC8GFVk)


### Result
The model achieved substantial accuracy on the dataset but as noted above, in the wild detection proved to be more challenging due to lack of data, and lack of variety in data.

[![Model-Accuracy.png](https://i.postimg.cc/QN9sBSvk/Model-Accuracy.png)](https://postimg.cc/SjhH3LSJ)