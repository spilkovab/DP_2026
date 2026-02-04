from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
model = YOLO("runs/detect/model_H2/weights/best.pt")
model.to("cuda")

model.model.half()

video_path = "vidz/palacak_08.MOV"
# cap = cv2.VideoCapture(video_path)

results = model.track(
    source=video_path,
    persist=True,
    stream=True,
    imgsz=640,
    device=0,
    half=True,
    vid_stride=2
)

for r in results:
    frame = r.plot()
    cv2.imshow("model G4 tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
# cap.release()
cv2.destroyAllWindows()

