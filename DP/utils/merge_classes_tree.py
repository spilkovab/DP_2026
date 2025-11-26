import os

def relabel_to_tree(data_dir):
    for split in ["train", "valid", "test"]:
        folder = os.path.join(data_dir, split, "labels")
        for file in os.listdir(folder):
            if not file.endswith(".txt"):
                continue
            path = os.path.join(folder,file)
            with open(path,"r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 5:
                    continue
                parts[0] = "0"
                new_lines.append(" ".join(parts) + "\n")
            with open(path,"w") as f:
                f.writelines(new_lines)
    print("all classes updated")

if __name__ == "__main__":
    path = "/home/student/Desktop/spilkova/dataset/trees.v6-final-dataset.yolov11"
    relabel_to_tree(path)