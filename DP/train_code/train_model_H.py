''' SCRIPT FOR TRAINING A YOLO MODEL'''
# TRAINED ON PALACAK 01, 02, 04
# import yolo
from ultralytics import YOLO

if __name__ == "__main__":
    # load pretrained yolo model
    model = YOLO('yolo11n.pt')

    # train model -- BARO ALWAYS CHANGE NAME FOR A NEW MODEL!!!
    model.train(
        # EDIT ALWAYS
        data='/home/student/Desktop/spilkova/dataset/data_drone_01/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        # EDIT ALWAYS
        name='model_H',  
        pretrained=True,
        multi_scale=True,
        patience=10,
        auto_augment='AugMix',
        visualize=True
    )

    # validation
    metrics = model.val()
    print(metrics)