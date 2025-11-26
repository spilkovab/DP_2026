'''
This is a function for extracting frames from videos
'''
import cv2
import os

# Input video file
name = "palacak_08"
video_path = "/home/student/Desktop/spilkova/vidz/" + name + ".MOV"
output_folder = "/home/student/Desktop/spilkova/dataset/" + name
frame_interval = 0.5  # Seconds

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# print statement
print("Extracting frames from video: " + name)

# Open the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) 
frame_skip = int(fps * frame_interval)  # Number of frames to skip

frame_count = 0
image_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    if frame_count % frame_skip == 0:
        image_path = os.path.join(output_folder, f"frame_{image_count:04d}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved: {image_path}")
        image_count += 1

    frame_count += 1

cap.release()
print(f"Frame extraction complete. Frames saved in: {video_path}.")
