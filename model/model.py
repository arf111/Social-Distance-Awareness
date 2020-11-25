import torch
import cv2
import random
from PIL import Image

class Model:
    """Model of Yolov5
    @ frame : frame from input stream 
	@ yolo_v : name of yolov5 version (s=small, m=medium, l=large, x=extra large)
    """
    def __init__(self, frame, yolo_v):
        self.frame = frame
        self.yolo_v = yolo_v
        # detect person only
        self.model = torch.hub.load('ultralytics/yolov5', yolo_v, pretrained=True, classes=1)
        self.results = model(frame[:, :, ::-1])

    def getCentroids(self):
        """Get centroids of the detected persons"""
        return results.xywh

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def getFrameBbox(self):
        """Get frame with boundary box"""
        
        return 

