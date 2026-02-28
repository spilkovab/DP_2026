from ultralytics import YOLO

# Load a model
model = YOLO("/home/student/Desktop/spilkova/runs/detect/model_J/weights/best.pt")  # load an official model

# Validate the model
metrics = model.val(project='/home/student/Desktop/spilkova/val_results/model_J/')  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list containing mAP50-95 for each category