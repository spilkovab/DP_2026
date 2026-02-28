''' SCRIPT FOR TRAINING A YOLO MODEL'''
# TRAINED ON PALACAK 01, 02, 04
# import yolo
from ultralytics import YOLO

#---------------------------------
DATA = 'data_05'
MODEL_NAME = 'model_J'
#---------------------------------

if __name__ == "__main__":
    # load pretrained yolo model
    model = YOLO('yolo11s.pt')

    # train model -- BARO ALWAYS CHANGE NAME FOR A NEW MODEL!!!
    model.train(
        # EDIT ALWAYS
        data=f'/home/student/Desktop/spilkova/dataset/{DATA}/data.yaml',
        epochs=150,
        imgsz=640,
        batch=8,
        # EDIT ALWAYS
        name=MODEL_NAME,  
        pretrained=True,
        multi_scale=True,
        patience=150,
        auto_augment='AugMix',
        visualize=True
    )

    # validation
    # metrics = model.val()
    # print(metrics)