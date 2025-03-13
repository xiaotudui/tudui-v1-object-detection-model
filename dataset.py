import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class VOCDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, num_classes=20):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.num_classes = num_classes
        self.img_filenames = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # VOC类别名称到索引的映射
        self.class_mapping = {
            'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
            'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
            'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
            'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
        }

    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        # 初始化标签
        class_label = torch.zeros(self.num_classes)
        bbox = torch.zeros(4)  # [x, y, w, h]
        
        # 获取第一个目标（如果存在多个目标，这里只取第一个）
        obj = root.find('object')
        if obj is not None:
            class_name = obj.find('name').text
            if class_name in self.class_mapping:
                class_idx = self.class_mapping[class_name]
                class_label[class_idx] = 1.0
                
                # 获取边界框坐标
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 转换为YOLO格式 [x_center, y_center, width, height]
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                bbox = torch.tensor([x_center, y_center, w, h])
        
        return class_label, bbox

    def __getitem__(self, index):
        # 读取图像
        img_filename = self.img_filenames[index]
        img_path = os.path.join(self.image_folder, img_filename)
        image = Image.open(img_path).convert('RGB')

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        # 读取XML标签文件
        xml_filename = os.path.splitext(img_filename)[0] + ".xml"
        xml_path = os.path.join(self.label_folder, xml_filename)

        # 解析XML文件获取类别和边界框
        class_label, bbox = self.parse_voc_xml(xml_path)
        
        # 组合标签
        target = torch.cat([class_label, bbox])

        return image, target

    def __len__(self):
        return len(self.img_filenames)


if __name__ == '__main__':
    dataset = VOCDataset(
        image_folder=r"../dataset/VOCdevkit/VOC2007/JPEGImages",
        label_folder=r"../dataset/VOCdevkit/VOC2007/Annotations"
    )
    print(dataset[0])
