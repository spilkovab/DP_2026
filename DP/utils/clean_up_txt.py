import os

def clean_list(file_path):
    if not os.path.exists(file_path):
        print(f"file not found: {file_path}")
        return

    with open(file_path, "r") as f:
        lines = f.read().splitlines()

    valid_lines = []

    for line in lines:
        img_path = line.strip()

        if img_path.startswith("data/"):
            img_path = img_path.replace("data/","DP1/",1)


        print(img_path)

        if os.path.exists(img_path):
            valid_lines.append(img_path)
        else:
            print(f"removed: {img_path}")

    with open(file_path, "w") as f:
        f.write("\n".join(valid_lines))


clean_list("DP1/Train.txt")
clean_list("DP1/Validation.txt")
