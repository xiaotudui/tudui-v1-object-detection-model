import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.img_filenames = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __getitem__(self, index):
        # 读取图像
        img_filename = self.img_filenames[index]
        img_path = os.path.join(self.image_folder, img_filename)
        image = Image.open(img_path).convert('RGB')

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        # 读取标签文件
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_filename)

        # 读取YOLO格式的标签 [class_id, x_center, y_center, width, height]
        boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    if line.strip():
                        values = list(map(float, line.strip().split()))
                        boxes.append(values)

        # 如果没有标注框，创建一个空的标注
        if not boxes:
            boxes = torch.zeros((0, 5))
        else:
            boxes = torch.tensor(boxes)

        return image, boxes

    def __len__(self):
        return len(self.img_filenames)


if __name__ == '__main__':
    dataset = VOCDataset(image_folder=r"C:\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages",
                         label_folder=r"C:\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\YOLO")

    print(dataset[0])
