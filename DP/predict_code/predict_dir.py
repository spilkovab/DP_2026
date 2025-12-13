from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLO11n model
model = YOLO("/home/student/Desktop/spilkova/runs/detect/model_G3/weights/best.pt")

# Define path to directory containing images and videos for inference
source = "/home/student/Desktop/spilkova/dataset/palacak_08_test"

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects

for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Save results to disk
    r.save(filename=f"results_G3/results{i}.jpg")