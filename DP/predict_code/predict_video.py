import cv2
from ultralytics import YOLO

#--------------------------------------
MODEL_NAME = 'model_H2'
VIDEO_PATH = "vidz/palacak_08.MOV"
OUTPUT_PATH = f"/home/student/Desktop/spilkova/outputs/{MODEL_NAME}_inference.mp4"
# --------------------------------------

model = YOLO(f"runs/detect/{MODEL_NAME}/weights/best.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference
    results = model(frame, verbose=False)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

cap.release()
out.release()

print(f"Saved true-speed inference video to: {OUTPUT_PATH}")