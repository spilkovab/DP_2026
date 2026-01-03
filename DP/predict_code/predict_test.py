from ultralytics import YOLO
import yaml

#---------------------------------
MODEL = 'model_G4'
YAML_FILE = 'data_02'
#---------------------------------

# Load model
model = YOLO(f"/home/student/Desktop/spilkova/runs/detect/{MODEL}/weights/best.pt")

# Validate model
metrics = model.val(
    data=f"/home/student/Desktop/spilkova/dataset/{YAML_FILE}/data.yaml",
    split="val",        # IMPORTANT (val/test)
    imgsz=640,
    batch=8,
    save_json=True,      # COCO-style results
    save_txt=True,       # YOLO-format predictions
    save_conf=True,
    plots=True,
    name=f"val_{MODEL}"
)

print(metrics)

# This needs to be done because yolo can parse yaml only for validation and training (BRUH)
with open(f"/home/student/Desktop/spilkova/dataset/{YAML_FILE}/data.yaml", "r") as f:
    data = yaml.safe_load(f)

test_path = f"/home/student/Desktop/spilkova/dataset/{YAML_FILE}/{data["test"]}"

# Run inference
results = model.predict(
    source=test_path,
    imgsz=640,
    save=True,
    conf=0.25,
    plots=True,
    name=f"predict_{MODEL}"
)

