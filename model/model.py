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
        # detect person only
        # self.model = create(name=self.yolo_v, pretrained=True, channels=3, classes=80).fuse().autoshape()
        self.model = torch.hub.load('ultralytics/yolov5', yolo_v, pretrained=True).fuse().autoshape()
        self.results = self.model(frame[:, :, ::-1])
        self.centroids = self.results.xywh[0][:,:] # tensor

    def getFrameBbox(self):
        """Get frame with boundary box"""
        points = self.results.pred[0][:,:] # tensor
        img = self.frame
        
        for i in range(points.shape[0]):
            if points[i][4] >= 0.55 and int(points[i][5]) == 0:
                img = cv2.rectangle(img,(int(points[i][0]), int(points[i][1])),
                    (int(points[i][2]), int(points[i][3])), (0, 0, 255), 2)
                img = cv2.circle(img, (int(self.centroids[i][0]), int(self.centroids[i][1])), 3, (0,255,0), 3)
                img = cv2.putText(img, 'person', (int(points[i][0]), int(points[i][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 
        
        return img

