import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import sys
import os

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.visualization import draw_custom_annotations

# Config
model_name = 'model_J'
video_path = "vidz/palacak_08_cut.MP4"
model_path = f"runs/detect/{model_name}/weights/best.pt"
save_path_video = f"/home/student/Desktop/spilkova/DP/track_results/annotated_tracking_{model_name}.mp4"
save_path_graph = f"/home/student/Desktop/spilkova/DP/track_results/trajectory_graph_{model_name}.mp4"

CLASS_COLOR = {
    0: (49, 211, 0),
    1: (255, 255, 0),
    2: (128, 0, 128),
    3: (0, 0, 255),
    4: (203, 192, 255)
}

MARGIN = 100  # Pixels for labels outside the main box

def get_rotated_text_image(text, font_scale, thickness, color):
    """Creates a small image with rotated text for the Y-axis."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # White backgropund
    txt_img = np.ones((h + 10, w + 10, 3), dtype=np.uint8) * 255
    cv2.putText(txt_img, text, (5, h + 2), font, font_scale, color, thickness)
    # Rotate 90 degrees counter-clockwise
    return cv2.rotate(txt_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def create_base_canvas(width, height, margin):
    """Draws the empty graph with boundaries and external labels."""
    canvas_w = width + (margin * 2)
    canvas_h = height + (margin * 2)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    color = (0, 0, 0)
    # Main tracking box
    cv2.rectangle(canvas, (margin, margin), (margin + width, margin + height), color, 2)
    
    # X axis
    x_text = f"Width: {width}px"
    cv2.putText(canvas, x_text, (margin + width // 2 - 80, margin + height + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    # Y axis
    y_text_img = get_rotated_text_image(f"Height: {height}px", 1.2, 2, color)
    rh, rw = y_text_img.shape[:2]
    y_offset = margin + (height // 2) - (rh // 2)
    canvas[y_offset : y_offset + rh, 20 : 20 + rw] = y_text_img
    
    return canvas

# INIT MODEL
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Video Writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(save_path_video, fourcc, fps, (orig_width, orig_height))
out_graph = cv2.VideoWriter(save_path_graph, fourcc, fps, (orig_width + MARGIN*2, orig_height + MARGIN*2))

# Display windows
display_width = 1000
display_height = int(orig_height * (display_width / orig_width))
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Graph", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", display_width, display_height)
cv2.resizeWindow("Graph", display_width, display_height)

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Fresh canvas every 50 frames
    current_graph = create_base_canvas(orig_width, orig_height, MARGIN)

    # YOLO Tracking
    results = model.track(frame, persist=True, verbose=False)[0]
    annotated_frame = frame.copy()

    if results.boxes is not None and results.boxes.id is not None:
        # Draw custom annotations
        annotated_frame = draw_custom_annotations(frame.copy(), results, names=model.names, mode='Name')
        
        boxes = results.boxes.xywh.cpu()
        track_ids = results.boxes.id.int().cpu().tolist()
        clss = results.boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

            if len(track) > 100:
                track.pop(0)

            # Draw graph
            color = CLASS_COLOR.get(int(cls), (0, 0, 0))
            
            # Convert track points to numpy array, add the margin offset
            pts = np.array(track).astype(np.int32)
            pts[:, 0] += MARGIN
            pts[:, 1] += MARGIN
            
            # Draw trail
            cv2.polylines(current_graph, [pts.reshape((-1, 1, 2))], isClosed=False, color=color, thickness=2)
            
            # Draw current position: dot + Object ID
            curr_x, curr_y = pts[-1]
            cv2.circle(current_graph, (curr_x, curr_y), 5, color, -1)
            cv2.putText(current_graph, f"ID:{track_id}", (curr_x + 8, curr_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Save videos
    out_video.write(annotated_frame)
    out_graph.write(current_graph)

    # Display results
    cv2.imshow("Video", annotated_frame)
    cv2.imshow("Graph", current_graph)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out_video.release()
out_graph.release()
cv2.destroyAllWindows()
print(f"Tracking complete. Files saved: \n1. {save_path_video}\n2. {save_path_graph}")
