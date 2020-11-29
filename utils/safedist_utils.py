import numpy as np
import cv2
import itertools
import math

def checkIfReg(center, W, H):
    if center[0][0] > W or center[0][0] < 0:
        return False
    elif center[1][0] > W or center[1][0] < 0:
        return False
    elif center[0][1] > H or center[0][1] < 0:
        return False 
    elif center[1][1] > H or center[1][1] < 0:
        return False
    return True

def get_Centroids_Lis(points, centroids, conf_score):
    centroids_lis = list()

    # Draw the detected boundary box
    for i in range(points.shape[0]):
        if points[i][4] >= conf_score and int(points[i][5]) == 0:
            centroids_lis.append(
                [centroids[i][0], centroids[i][1]])

    return np.float32(np.array(centroids_lis)).reshape(-1, 1, 2)

def draw_RiskLine(img, bird_view, minm_dist, centroids_arr, transformed_centroids, warped_imgH, warped_imgW):
    """Find out the violation of social distance measure and return the frame with those points 
    Arguments:
    @ img: frame
    @ minm_dist: minimum threshold of distance (6 ft)
    @ transformed_centroids: center of detected boundary box
    """
    if len(transformed_centroids) >= 2:
    # Iterate over every possible 2 by 2 between the points combinations 
        for i in range(len(transformed_centroids)-1):
            for j in range(len(transformed_centroids)):
                if i!=j:
                    pair = [[transformed_centroids[i][0], transformed_centroids[i][1]], 
                        [transformed_centroids[j][0], transformed_centroids[j][1]]]
                    if checkIfReg(pair, warped_imgW, warped_imgH): 
                    # Check if the distance between each combination of points is less than the minimum distance chosen
                        if int(math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 )) < int(minm_dist): 
                            img = cv2.line(img, (int(centroids_arr[i][0][0]), int(centroids_arr[i][0][1])), 
                                    (int(centroids_arr[j][0][0]), int(centroids_arr[j][0][1])), color=(0,0,255), thickness=2) # red line
                            bird_view = cv2.line(bird_view, (int(pair[0][0]), int(pair[0][1])), 
                                    (int(pair[1][0]), int(pair[1][1])), color=(0,0,255), thickness=2)
    
def show_Detection(img, bird_view, centroids_arr, transformed_centroids, det_points, warped_imgH, warped_imgW):
    """Draw the detected boxes and centroids inside the designated region.
    Arguments:
    @ img: frame
    @ centroids_lis: list of detected centroids
    @ transformed_centroids: warped centroids
    @ det_points: detected boundary boxes, conf score, class
    @ warped_imgH: warped image's height
    @ warped_imgW: warped image's width

    """
    if len(transformed_centroids) >= 2:
        for i in range(len(transformed_centroids)):
            # check if transformed_centroids are in the designated region
            if not ((transformed_centroids[i][0] > warped_imgW or transformed_centroids[i][0] < 0) or (transformed_centroids[i][1] > warped_imgH or transformed_centroids[i][1] < 0)):
                # Define the detections as circle in bird view image.
                bird_view = cv2.circle(bird_view, (int(transformed_centroids[i][0]), int(transformed_centroids[i][1])), 3, (0,255,0), 3)
                
                # Draw the Bboxes, centroids, and write down class name.
                img = cv2.rectangle(img,(int(det_points[i][0]), int(det_points[i][1])),
                    (int(det_points[i][2]), int(det_points[i][3])), (140, 102, 39), 2)
                img = cv2.circle(img, (int(centroids_arr[i][0][0]), int(centroids_arr[i][0][1])), 3, (0,255,0), 3)
                img = cv2.putText(img, 'person', (int(det_points[i][0]), int(det_points[i][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 