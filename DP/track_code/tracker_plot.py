from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
# Path to utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.visualization import draw_custom_annotations

# ------------- LOAD PATHS -------------
video_path = "vidz/palacak_08.MOV"
model_name = 'model_I'
project_path = "DP/track_results"
# --------------------------------------

# Load the YOLO model
model = YOLO(f"runs/detect/{model_name}/weights/best.pt")

CLASS_COLOR = {
    0: (49, 211,0),
    1: (255,255,0),
    2: (128,0,128),
    3: (0,0,255),
    4: (203,192,255)
}

# load video
cap = cv2.VideoCapture(video_path)

# get size for windows
orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
display_width = 1000
display_height = int(orig_height * (display_width / orig_width))

# Windows names
win_video = "Custom Visuals (Video)"
win_graph = "Trajectory Map (White)"

# windows init
cv2.namedWindow(win_video, cv2.WINDOW_NORMAL)
cv2.namedWindow(win_graph, cv2.WINDOW_NORMAL)

# set window size
cv2.resizeWindow(win_video, display_width, display_height)
cv2.resizeWindow(win_graph, display_width, display_height)

# place windows next to each other
cv2.moveWindow(win_video, 50, 100)
cv2.moveWindow(win_graph, display_width + 70, 100)

trajectory_map = np.ones((orig_height, orig_width, 3), dtype=np.uint8) * 255

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # read frame
    success, frame = cap.read()

    if success:
        # persisting tracks between frames
        result = model.track(frame, persist=True)[0]

        # get the boxes and track IDs
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            clss = result.boxes.cls.int().cpu().tolist()

            # custom visualization
            annotated_frame = draw_custom_annotations(result.orig_img.copy(), result, names=model.names, mode='Name')

            # plot tracks
            for box, track_id, clas in zip(boxes, track_ids,clss):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)
                

                # Draw the tracking lines
                color = CLASS_COLOR.get(int(clas), (0, 0, 0))
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(trajectory_map, [points], isClosed=False, color=color, thickness=2)

                # trajectory id
                start_x, start_y = track[0]
                # cv2.putText(trajectory_map, f"ID: {track_id}", (int(start_x), int(start_y)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        else:
            annotated_frame = frame

        # Display the annotated frame
        cv2.imshow(win_video, annotated_frame)
        cv2.imshow(win_graph, trajectory_map)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()