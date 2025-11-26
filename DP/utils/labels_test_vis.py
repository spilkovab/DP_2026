# from ultralytics.data.utils import YOLODataset

# dataset = YOLODataset("/home/student/Desktop/spilkova/dataset/DP1/images/Train/", "/home/student/Desktop/spilkova/dataset/DP1/labels")
# dataset.visualize("output_folder")

from ultralytics.utils.plotting import plot_labels
from ultralytics.utils.plotting import save_one_box
import cv2
import os

images = [
    "/home/student/Desktop/spilkova/dataset/DP1/images/Train/frame_0004.jpg",
    "/home/student/Desktop/spilkova/dataset/DP1/images/Train/frame_0004_1.jpg",
    "/home/student/Desktop/spilkova/dataset/DP1/images/Train/frame_0004_2.jpg",
    "/home/student/Desktop/spilkova/dataset/DP1/images/Train/frame_0005_2.jpg"
]

labels_dir = "/home/student/Desktop/spilkova/dataset/DP1/labels/Train"
save_dir = "/home/student/Desktop/spilkova/teset_labels_vis_01"
os.makedirs(save_dir, exist_ok=True)

for img_path in images:
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    label_path = os.path.join(
        labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    )

    if not os.path.exists(label_path):
        print("No label:", label_path)
        continue

    with open(label_path) as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.split())

            # convert normalized → absolute pixel coords
            x *= w
            y *= h
            bw *= w
            bh *= h

            x1 = int(x - bw / 2)
            y1 = int(y - bh / 2)
            x2 = int(x + bw / 2)
            y2 = int(y + bh / 2)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label text
            cv2.putText(
                img,
                str(int(cls)),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    out = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(out, img)
    print("Saved:", out)
