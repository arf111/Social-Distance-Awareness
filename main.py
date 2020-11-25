import cv2
import numpy as np
import argparse
import utils

mouse_pts = []

image = None
np.random.seed(42)

def get_Points(event, x, y, flags, params):
    """Get the points from opencv's setMouseCallBack function."""
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pts.append((x,y))

        if len(mouse_pts) <= 4:
            cv2.circle(image, (x,y), 5, (0,255,0), 5)
        else:
            cv2.circle(image, (x,y), 5, (255,0,0), 5)

        if len(mouse_pts) > 1 and len(mouse_pts) <= 4:
            cv2.line(image, (x,y), (mouse_pts[len(mouse_pts)-2][0], mouse_pts[len(mouse_pts)-2][1]), (0,0,0), 2)

            if len(mouse_pts) == 4:
                cv2.line(image, (x,y), (mouse_pts[0][0], mouse_pts[0][1]), (0,0,0), 2)
        

def calc_dist(inp_vid, out_vid_path):
    """Calculate the distance between peoples in a frame."""

    cap = cv2.VideoCapture(inp_vid)

    inp_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    inp_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    inp_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_vid_path+"/result.avi", fourcc, inp_fps,  (inp_width, inp_height))

    first_frame = True

    global image

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            print("Frame not found")
            break

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

            first_frame = False

        # ordering => (bleft,tleft,tright,bright)
        # The points must be drawn in this order.
        src_points = np.float32(np.array(mouse_pts[:4]))
        dest_points = np.float32([[0, H], [0, 0], [W, 0], [W, H]])
        pers_matrix = cv2.getPerspectiveTransform(src_points, dest_points)

        # User-defined points to get safe distance (approx 6 ft)
        safe_points = np.float32(np.array([mouse_pts[4:6]])) # perspectiveTransform expects 2D/3D channel of floating point array where each element is 2D/3D vector.
        warped_safe_points = cv2.perspectiveTransform(safe_points, pers_matrix)[0]

        # From user-defined, we'll get the safe distance (approx 6 ft). Euclidean distance
        safe_dist = np.sqrt((warped_safe_points[0][0] - warped_safe_points[1][0])**2 + (
            warped_safe_points[0][1] - warped_safe_points[1][1]) ** 2)
        
        # Draw the rectangle in the image
        cv2.polylines(frame, [np.array(mouse_pts[:4])], True, (0,0,0), 2)

        cv2.imwrite("out.jpg",frame)
        break
        
        # --------------------------YOLOv5-------------------------#
        # Get the centroids and bbox visualizer
        
        
        # Press 'q' for exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        

if __name__== "__main__":
    # Receives arguments specified by user
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', default='./testing/TRIDE.mp4', help='Path for input video')
                    
    parser.add_argument('--output_dir', default='./output/', help='Path for output video')
    
    options = parser.parse_args()

    inp_vid = options.input_path
    out_vid_path = options.output_dir

    cv2.namedWindow("image")

    cv2.setMouseCallback("image",get_Points)

    calc_dist(inp_vid, out_vid_path)