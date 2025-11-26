from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="DP/DP1/data.yaml", epochs=1, imgsz=640)
