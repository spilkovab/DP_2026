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
MODEL_NAME = 'model_J'
VIDEO_PATH = "vidz/palacak_08.MOV"
OUTPUT_PATH = f"/home/student/Desktop/spilkova/outputs/{MODEL_NAME}_inference_dobratice_02.mp4"
# --------------------------------------

# load model
model = YOLO(f"runs/detect/{MODEL_NAME}/weights/best.pt")
# load video
cap = cv2.VideoCapture(VIDEO_PATH)

# get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    # read frame
    success, frame = cap.read()

    if success:
        # run inference
        results_genereator = model(frame, stream=True)

        for results in results_genereator:
            # custom visualization
            annotated_frame = draw_custom_annotations(frame, results[0], names=model.names, mode='ID')
            # annotated_frame = results[0].plot()

            # show frame
            cv2.imshow("YOLO Inference", annotated_frame)

        # exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # break on the last frame
        break


cap.release()
cv2.destroyAllWindows()