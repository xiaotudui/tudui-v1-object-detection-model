import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, num_classes=20):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.num_classes = num_classes
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

        # 初始化one-hot编码和边界框
        class_label = torch.zeros(self.num_classes)
        bbox = torch.zeros(4)  # [x, y, w, h]

        if os.path.exists(label_path):
            with open(label_path) as f:
                line = f.readline().strip()
                if line:
                    values = list(map(float, line.split()))
                    class_id = int(values[0])
                    class_label[class_id] = 1.0  # one-hot编码
                    bbox = torch.tensor(values[1:])  # 边界框坐标

        # 组合标签
        target = torch.cat([class_label, bbox])

        return image, target

    def __len__(self):
        return len(self.img_filenames)


if __name__ == '__main__':
    dataset = VOCDataset(image_folder=r"C:\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages",
                         label_folder=r"C:\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\YOLO")

    print(dataset[0])
