# Social-Distance-Awareness
Project measuring if people follow the appropriate distance between themselves in order to prevent dispersement of COVID-19 disease.

- Bird eye view is implemented to precisely measure the distance between detected objects.
- For detection, Yolov5 model is used.

<img src='imgs/SDAware.gif'>

## Prerequisites

Python 3.6 or later with dependencies [requirements.txt](requirements.txt) installed. Run:

```
$ pip install -r requirements.txt
```

## Getting Started

To run the program, perform:

```
$ python main.py
```

To use different model of Yolov5 use yolov5s for small, yolov5m for medium, or yolov5l for large model. For example:

```
$ python main.py --yolov yolov5m
```

## Usage


The following steps occur when user runs the [main.py](main.py) file:

1. First of all, a frame will be given of the input video to define the boundary region where the detection will occur. The user must click 4 points in the frame in the order of bottom-left, top-left, top-right, and bottom-right. Note that this ordering is really important to accurately find out the distances between people. The 4 points must form a rectangle in the designated region. 
2. After selecting 4 points, 2 extra points are needed from the user to define the approx. 6 ft. distance in the frame. This will be user-defined.
3. Finally, the violated distances will be shown in original frame in accordance with the bird view frame. Yolov5 is used to detect each person.

## References

1. Perspective Transformation: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
2. Yolov5: https://github.com/ultralytics/yolov5