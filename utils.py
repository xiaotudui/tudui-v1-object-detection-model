import torch
import numpy as np
from torchvision import transforms


def intersection_over_union(boxes_preds, boxes_labels):
    """
    计算预测框和真实框的IOU
    boxes_preds shape: (N, 4) where N是预测框数量，4代表(x_center, y_center, width, height)
    boxes_labels shape: (N, 4)
    """
    # 获取框的坐标
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # 计算交集区域的坐标
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # 计算交集面积，需要处理没有交集的情况
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # 计算各自的面积
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # 计算IOU
    union = box1_area + box2_area - intersection + 1e-6
    iou = intersection / union

    return iou


def get_transforms(train=True):
    """
    获取数据转换
    """
    if train:
        return transforms.Compose([
            transforms.Resize((418, 418)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((418, 418)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



