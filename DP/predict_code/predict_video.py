import cv2
from ultralytics import YOLO

#--------------------------------------
MODEL_NAME = 'model_I'
VIDEO_PATH = "vidz/dobratice_02.MOV"
OUTPUT_PATH = f"/home/student/Desktop/spilkova/outputs/{MODEL_NAME}_inference_palacak_08.mp4"
# --------------------------------------

model = YOLO(f"runs/detect/{MODEL_NAME}/weights/best.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # Run inference
#     results = model(frame, verbose=False, show=True)

#     # Annotate frame
#     annotated_frame = results[0].plot()

#     # Save frame
#     out.write(annotated_frame)

# cap.release()
# out.release()

# print(f"Saved true-speed inference video to: {OUTPUT_PATH}")