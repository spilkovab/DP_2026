import os

images_dir = "DP1/images/Validation"
labels_dir = "DP1/labels/Validation"

for img_file in os.listdir(images_dir):
    base_name = os.path.splitext(img_file)[0]
    label_path = os.path.join(labels_dir, f"{base_name}.txt")
    # print(label_path)

    if not os.path.exists(label_path):
        img_path = os.path.join(images_dir, img_file)
        os.remove(img_path)
        print(img_path)

print('DONE')
