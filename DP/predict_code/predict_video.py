import cv2
import sys
import os
from ultralytics import YOLO

# Path for utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.visualization import draw_custom_annotations

#------------- LOAD PATHS --------------
MODEL_NAME = 'model_K3'
VIDEO_PATH = "/home/student/Desktop/spilkova/vidz/palacak_08.MOV"
OUTPUT_PATH = f"/home/student/Desktop/spilkova/outputs/{MODEL_NAME}_inference_palacak_08.mp4"
# --------------------------------------

# load model
model = YOLO(f"runs/detect/{MODEL_NAME}/weights/best.pt")
model.to('cuda') # to GPU
# load video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Cannot open video on path: {VIDEO_PATH}")
    print(f"Absolute path: {os.path.abspath(VIDEO_PATH)}")
    sys.exit()

# get video properties
fps = cap.get(cv2.CAP_PROP_FPS)

target_width = 1280
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = target_width / original_width
target_height = int(original_height * scale)

# fallback
if fps == 0: fps = 30 

print(f"Video loaded: {target_width}x{target_height} at {fps} FPS")

fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (target_width, target_height))

if not out.isOpened():
    print("ERROR: Couldn't create file.")


while cap.isOpened():
    # read frame
    success, frame = cap.read()

    if success:

        frame_resized = cv2.resize(frame, (target_width, target_height))
        # run inference
        results = model(frame_resized,imgsz=640,verbose=False)

        # custom visualization
        annotated_frame = draw_custom_annotations(frame_resized, results[0], names=model.names, mode='ID')
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        # show frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # break on the last frame
        break


cap.release()
out.release()
cv2.destroyAllWindows()