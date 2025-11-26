''' SCRIPT FOR TRAINING A YOLO MODEL'''
# TRAINED ON PALACAK 01, 02, 04
# import yolo
from ultralytics import YOLO

if __name__ == "__main__":
    # load pretrained yolo model
    model = YOLO('yolo11s.pt')

    # train model -- BARO ALWAYS CHANGE NAME FOR A NEW MODEL!!!
    model.train(
        data='/home/student/Desktop/spilkova/DP/DP1/data.yaml',
        epochs=150,
        imgsz=640,
        batch=8,
        name='model_D',  
        pretrained=True,
        multi_scale=True,
        hsv_h=0.015,
        hsv_v=0.3,
        hsv_s=0.5,
        translate=0.1,
        scale=0.5,
        shear=0.2,
        perspective=0.0005,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,
        erasing=0.4,
        patience=10,
        lr0=0.1,
        lrf=0.1
    )

    # validation
    metrics = model.val()
    print(metrics)