''' SCRIPT FOR TRAINING A YOLO MODEL'''
# import yolo
from ultralytics import YOLO

#---------------------------------
DATA = 'data_06'
MODEL_NAME = 'model_K'
#---------------------------------

if __name__ == "__main__":
    # load pretrained yolo model
    model = YOLO('yolo11s.pt')

    # train model -- BARO ALWAYS CHANGE NAME FOR A NEW MODEL!!!  
    model.train(
        data=f'/home/student/Desktop/spilkova/dataset/{DATA}/data.yaml',
        imgsz=640,
        batch=8,
        name=MODEL_NAME,  
        pretrained=True,
        multi_scale=True,
        patience=150,
        auto_augment='AugMix',
        visualize=True
    )

    # validation
    metrics = model.val()
    print(metrics)
