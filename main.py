import torch
import cv2
import numpy as np
import argparse
from utils.safedist_utils import draw_RiskLine, show_Detection, get_Centroids_Lis

mouse_pts = []

image = None
np.random.seed(42)

def get_Points(event, x, y, flags, params):
    """Get the points from opencv's setMouseCallBack function."""
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pts.append((x, y))

        if len(mouse_pts) <= 4:
            cv2.circle(image, (x, y), 5, (0, 255, 0), 5)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 5)

        if len(mouse_pts) > 1 and len(mouse_pts) <= 4:
            cv2.line(image, (x, y), (mouse_pts[len(
                mouse_pts)-2][0], mouse_pts[len(mouse_pts)-2][1]), (0, 0, 0), 2)

            if len(mouse_pts) == 4:
                cv2.line(image, (x, y),
                         (mouse_pts[0][0], mouse_pts[0][1]), (0, 0, 0), 2)

def calc_dist(inp_vid, out_vid_path, yolo_v, conf_score):
    """Calculate the distance between peoples in a frame. Shows the result in a frame, and generates a video.
    @ inp_vid: input video path
    @ out_vid_path: output video path
    @ yolo_v: yolov version
    @ conf_score: confidence score of detection
    """

    cap = cv2.VideoCapture(inp_vid)

    inp_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_vid_path+"result.avi",
                          fourcc, inp_fps,  (int(cap.get(3)), int(cap.get(4))))

    model = torch.hub.load('ultralytics/yolov5', yolo_v, pretrained=True).fuse().autoshape()

    first_frame = True

    global image

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break
        else:
            # Get height and width of original frame.
            (H, W) = frame.shape[0], frame.shape[1]

            # Set the ROI in the first frame of the input video.
            if first_frame:
                image = frame.copy()

                while True:
                    cv2.imshow("image", image)
                    cv2.waitKey(1)
                    if len(mouse_pts) == 7:
                        cv2.destroyWindow("image")
                        break
            
            # ordering => (bleft, tleft, tright, bright). The points must be drawn in this order.
            src_points = np.float32(np.array(mouse_pts[:4]))
            dest_points = np.float32([[0, H], [0, 0], [W, 0], [W, H]])
            
            # get the perspective matrix and warped image's height and width.
            pers_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
            warped_img = cv2.warpPerspective(frame, pers_matrix, (W, H))
            warped_imgH, warped_imgW = warped_img.shape[0], warped_img.shape[1] 

            # bird eye view
            bird_view = np.zeros((warped_imgH, warped_imgW, 3), np.uint8)
            # bird eye view color background 190, 191, 184
            bird_view[:] = (184, 191, 190)
            
            # User-defined points to get safe distance (approx 6 ft)
            # perspectiveTransform expects 2D/3D channel of floating point array where each element is 2D/3D vector.
            safe_points = np.float32(np.array([mouse_pts[4:6]]))
            warped_safe_points = cv2.perspectiveTransform(
                safe_points, pers_matrix)[0]

            # From user-defined, we'll get the safe distance (approx 6 ft). Euclidean distance
            safe_dist = np.sqrt((warped_safe_points[0][0] - warped_safe_points[1][0])**2 + (
                warped_safe_points[0][1] - warped_safe_points[1][1]) ** 2)

            # Draw the designated region in the image
            cv2.polylines(frame, [np.array(mouse_pts[:4])], True, (0, 0, 0), 2)
            
            # --------------------------YOLOv5-------------------------#
            
            # Get the results from the model.
            results = model(frame[:, :, ::-1])
            
            # detected bboxes with confidence score and class.
            det_points = results.pred[0][:,:]
            centroids = results.xywh[0][:,:2]
            
            # transform the centroids.
            centroids_arr = get_Centroids_Lis(det_points, centroids, conf_score)
            warped_centroids = cv2.perspectiveTransform(centroids_arr, pers_matrix)

            # make a list of the warped centroids
            warped_centroids_lis = list()
            for i in range(warped_centroids.shape[0]):
                warped_centroids_lis.append(
                    [warped_centroids[i][0][0], warped_centroids[i][0][1]])
            
            # Show detection on main frame and draw line.
            show_Detection(frame, bird_view, centroids_arr, warped_centroids_lis, det_points, warped_imgH, warped_imgW)
            draw_RiskLine(frame, bird_view, safe_dist, centroids_arr, warped_centroids_lis, warped_imgH, warped_imgW)
            
            # Write to video and show image on window.
            out.write(frame)
            cv2.imshow('output', frame)
            cv2.imshow('bird_view', bird_view)

            # Write to bird view frame.
            if first_frame:
                fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
                out2 = cv2.VideoWriter(out_vid_path+"/bird_view.avi",
                          fourcc, inp_fps,  (bird_view.shape[1], bird_view.shape[0]))
            else:
                out2.write(bird_view)

            first_frame = False
            # Press 'q' for exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Receives arguments specified by user
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='./test/TRIDE.mp4', help='Path for input video')
    parser.add_argument('--output_dir', default='./output/', help='Path for output video')
    parser.add_argument('--yolov', default='yolov5m', help='Load yolov5 model. The models are yolov5s = small, yolov5m = medium, yolov5l = large.')
    parser.add_argument('--conf_score', default=0.70, help = 'Detection confidence score. Write as floating point value')

    options = parser.parse_args()

    inp_vid = options.input_path
    out_vid_path = options.output_dir
    yolo_v = options.yolov
    conf_score = options.conf_score

    cv2.namedWindow("image")

    cv2.setMouseCallback("image", get_Points)

    calc_dist(inp_vid, out_vid_path, yolo_v, conf_score)
