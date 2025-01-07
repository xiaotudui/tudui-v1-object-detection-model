import torch
import numpy as np
from torchvision import transforms


def get_transforms(train=True):
    """
    获取数据转换
    """
    if train:
        return transforms.Compose([
            transforms.Resize((418, 418)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((418, 418)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



