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
from moviepy import VideoFileClip

# for debugging
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

# ------------- LOAD PATHS -------------
orig_video_path = "vidz/palacak_08.MOV"
video_path = "vidz/palacak_08_cut.MP4"
model_name = 'model_J'
project_path = "DP/track_results"
output_path = f"outputs/track_{model_name}_palacak_08.mp4"
# --------------------------------------

# CUT original video
if not os.path.exists(video_path):
    with VideoFileClip(orig_video_path) as video:
        # subclip
        new_video = video.subclipped(60, 105)
        
        # save
        new_video.write_videofile(video_path, codec="libx264", audio_codec="aac")


# load model
model = YOLO(f"runs/detect/{model_name}/weights/best.pt")

model.to("cuda")
model.model.half()

temp_cap = cv2.VideoCapture(video_path)
fps = temp_cap.get(cv2.CAP_PROP_FPS)
width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_height := temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
temp_cap.release()

# VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


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
    # cv2.imshow("model G4 tracking", frame)
    # save frame
    out.write(frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

# Release the video capture object and close the display window
# cap.release()
out.release()
cv2.destroyAllWindows()

