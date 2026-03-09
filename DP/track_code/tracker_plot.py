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
video_path = "vidz/palacak_08_cut.MP4"
model_name = 'model_K'
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


# 1. SETUP VIDEO SAVING
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
save_path_video = f"track_results/annotated_tracking_{model_name}.mp4"
save_path_graph = f"track_results/trajectory_graph_{model_name}.mp4"

out_video = cv2.VideoWriter(save_path_video, fourcc, fps, (orig_width, orig_height))
out_graph = cv2.VideoWriter(save_path_graph, fourcc, fps, (orig_width, orig_height))

def get_rotated_text_image(text, font_scale, thickness, color):
    """Helper to create a rotated text surface."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Create a small white canvas for the text
    txt_canvas = np.ones((h + 10, w + 10, 3), dtype=np.uint8) * 255
    cv2.putText(txt_canvas, text, (5, h + 5), font, font_scale, color, thickness)
    return cv2.rotate(txt_canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)

def draw_labeled_canvas(width, height, margin):
    """Creates a canvas with labels outside the boundary box."""
    canvas = np.ones((height + margin*2, width + margin*2, 3), dtype=np.uint8) * 255
    color = (0, 0, 0)
    
    # 1. DRAW BOUNDARY (The actual image area)
    # Top-left of box is at (margin, margin)
    cv2.rectangle(canvas, (margin, margin), (margin + width, margin + height), color, 2)
    
    # 2. X-AXIS LABEL (Bottom)
    x_label = f"Width: {width}px"
    cv2.putText(canvas, x_label, (margin + width//2 - 100, margin + height + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # 3. Y-AXIS LABEL (Left - Rotated)
    y_label_img = get_rotated_text_image(f"Height: {height}px", 0.8, 2, color)
    rh, rw = y_label_img.shape[:2]
    # Paste the rotated label into the left margin
    y_pos = margin + height//2 - rh//2
    canvas[y_pos : y_pos+rh, 10 : 10+rw] = y_label_img
    
    return canvas

# --- UPDATE VIDEO WRITER FOR LARGER CANVAS ---
out_graph = cv2.VideoWriter(save_path_graph, fourcc, fps, (display_width, display_height))

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. REFRESH GRAPH EVERY FRAME (for the 50-frame tail effect)
    current_graph = np.ones((orig_height, orig_width, 3), dtype=np.uint8) * 255
    draw_axes(current_graph, orig_width, orig_height)

    # Tracking inference
    result = model.track(frame, persist=True)[0]
    annotated_frame = result.orig_img.copy()

    current_graph = draw_labeled_canvas(orig_width, orig_height, MARGIN)

    if result.boxes and result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        clss = result.boxes.cls.int().cpu().tolist()

        # Custom visualization for the main video
        annotated_frame = draw_custom_annotations(annotated_frame, result, names=model.names, mode='Name')

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

            if len(track) > 50:
                track.pop(0)

            # 2. OFFSET THE TRACKING POINTS 
            # Because the box starts at (MARGIN, MARGIN)
            points = np.array(track).astype(np.int32)
            points[:, 0] += MARGIN
            points[:, 1] += MARGIN
            
            color = CLASS_COLOR.get(int(cls), (0, 0, 0))
            cv2.polylines(current_graph, [points.reshape((-1, 1, 2))], isClosed=False, color=color, thickness=3)
            
            # 3. CURRENT POSITION + ID LABEL
            curr_x, curr_y = int(x) + MARGIN, int(y) + MARGIN
            cv2.circle(current_graph, (curr_x, curr_y), 6, color, -1)
            cv2.putText(current_graph, f"ID {track_id}", (curr_x + 10, curr_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # If no tracks, just use the clean frame
        annotated_frame = frame

    # 4. WRITE FRAMES TO FILES
    out_video.write(annotated_frame)
    out_graph.write(current_graph)

    # Display (Resized for your screen)
    cv2.imshow(win_video, cv2.resize(annotated_frame, (display_width, display_height)))
    cv2.imshow(win_graph, cv2.resize(current_graph, (display_width, display_height)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 5. RELEASE EVERYTHING
cap.release()
out_video.release()
out_graph.release()
cv2.destroyAllWindows()
print(f"Videos saved: {save_path_video} and {save_path_graph}")

# # Loop through the video frames
# while cap.isOpened():
#     # read frame
#     success, frame = cap.read()

#     if success:
#         # persisting tracks between frames
#         result = model.track(frame, persist=True)[0]

#         # get the boxes and track IDs
#         if result.boxes and result.boxes.is_track:
#             boxes = result.boxes.xywh.cpu()
#             track_ids = result.boxes.id.int().cpu().tolist()
#             clss = result.boxes.cls.int().cpu().tolist()

#             # custom visualization
#             annotated_frame = draw_custom_annotations(result.orig_img.copy(), result, names=model.names, mode='Name')

#             # plot tracks
#             for box, track_id, clas in zip(boxes, track_ids,clss):
#                 x, y, w, h = box
#                 track = track_history[track_id]
#                 track.append((float(x), float(y)))  # x, y center point
#                 if len(track) > 30:  # retain 30 tracks for 30 frames
#                     track.pop(0)
                

#                 # Draw the tracking lines
#                 color = CLASS_COLOR.get(int(clas), (0, 0, 0))
#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(trajectory_map, [points], isClosed=False, color=color, thickness=2)

#                 # trajectory id
#                 start_x, start_y = track[0]
#                 # cv2.putText(trajectory_map, f"ID: {track_id}", (int(start_x), int(start_y)), 
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#         else:
#             annotated_frame = frame

#         # Display the annotated frame
#         cv2.imshow(win_video, annotated_frame)
#         cv2.imshow(win_graph, trajectory_map)


#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()