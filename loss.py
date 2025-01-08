import torch.nn as nn


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_criterion = nn.CrossEntropyLoss()  # 用于类别预测
        self.box_criterion = nn.MSELoss()  # 用于边界框预测

    def forward(self, predictions, targets):
        # 分离类别预测和边界框预测
        pred_classes = predictions[:, :20]  # 前20个是类别预测
        pred_boxes = predictions[:, 20:]    # 后4个是边界框预测
        
        target_classes = targets[:, :20]    # 目标类别（one-hot）
        target_boxes = targets[:, 20:]      # 目标边界框
        
        # 计算类别损失和边界框损失
        class_loss = self.class_criterion(pred_classes, target_classes)
        box_loss = self.box_criterion(pred_boxes, target_boxes)
        
        # 可以调整这些权重
        total_loss = class_loss + box_loss
        
        return total_loss