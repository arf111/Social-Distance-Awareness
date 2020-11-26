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
        self.model = torch.hub.load('ultralytics/yolov5', yolo_v, force_reload=True, pretrained=True).autoshape()
        self.results = model(frame[:, :, ::-1])
        self.centroids = results.xywh[0][:,:] # tensor

    def getFrameBbox(self):
        """Get frame with boundary box"""
        points = results.pred[0][:,:] # tensor

        for i in range(points.shape[0]):
            if points[i][4] >= 0.75 and int(points[i][5]) == 0:
                img = cv2.rectangle(img,(int(points[i][0]), int(points[i][1])),
                    (int(points[i][2]), int(points[i][3])), (0, 0, 255), 2)
                img = cv2.circle(img, (int(centroids[i][0]), int(centroids[i][1])), 5, (0,255,0), 5)
                # img = cv2.putText()
        
        return img

