from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import torch
import sys
import os
# Path to utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.visualization import draw_custom_annotations

# for debugging
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

# ------------- LOAD PATHS -------------
video_path = "vidz/dobratice_02.MOV"
model_name = 'model_I'
project_path = "DP/track_results"
# --------------------------------------

# load model
model = YOLO(f"runs/detect/{model_name}/weights/best.pt")

model.to("cuda")
model.model.half()

# run tracking
results = model.track(
    source=video_path,
    persist=True,
    stream=True,
    imgsz=640,
    device=0,
    half=True,
    vid_stride=2,
    save_txt=True,
    project=project_path
)

for r in results:
    # draw custom visualization
    frame = draw_custom_annotations(r.orig_img.copy(), r, names=model.names, mode='Name')
    # show frame
    cv2.imshow("model G4 tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
# cap.release()
cv2.destroyAllWindows()

