from ultralytics import YOLO

# Import model
model_names = ["model_F11", "model_G4", "model_H2", "model_I", "model_J", "model_K3"]
data_names = ["data", "data_02", "data_03", "data_04", "data_05", "data_06"]

for model_name, data_name in zip(model_names, data_names):
    model = YOLO(f"/home/student/Desktop/spilkova/runs/detect/{model_name}/weights/best.pt")
    data = f"/home/student/Desktop/spilkova/dataset/{data_name}/data.yaml"

    model.benchmark(data=data, imgsz=640, device=0)

    # # Run validation
    # metrics = model.val

    # # Get speed
    # avg_preprocess = metrics.speed['preprocess']
    # avg_inference = metrics.speed['inference']
    # avg_postprocess = metrics.speed['postprocess']

    # # Print results
    # print("--------------------------------------------------------------------------")
    # print(f"------------------------- SPEED {model_name} -----------------------------")
    # print(f"Average preprocess speed of model {model_name}: {avg_preprocess} \n")
    # print(f"Average inference speed of model {model_name}: {avg_inference} \n")
    # print(f"Average postprocess speed of model {model_name}: {avg_postprocess} \n")
    # print("--------------------------------------------------------------------------")

    # # Export model
    # model.Export(format='onnx')
    # onnx_model = YOLO('model.onnx')
    # # Simulation on a weaker engine ('cpu' argument)
    # results = onnx_model.val(device='cpu')