import torch.nn as nn


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()  # 类别的交叉熵损失
        self.mse = nn.MSELoss()  # 边界框的均方误差损失

        # 可以调整这两个权重来平衡分类误差和定位误差
        self.lambda_class = 1.0
        self.lambda_box = 1.0

    def forward(self, predictions, targets):
        """
        计算检测损失：类别损失 + 边界框损失

        Args:
            predictions: (batch_size, num_classes + 4)
                        前num_classes个是类别预测，后4个是边界框预测 [x, y, w, h]
            targets: (batch_size, 5)
                    [class_id, x, y, w, h]
        """
        # 1. 类别损失
        class_pred = predictions[:, :self.num_classes]  # 类别预测
        class_target = targets[:, 0].float()  # 目标类别
        class_loss = self.ce(class_pred, class_target)

        # 2. 边界框损失
        box_pred = predictions[:, self.num_classes:]  # 预测的边界框
        box_target = targets[:, 1:]  # 目标边界框
        box_loss = self.mse(box_pred, box_target)

        # 总损失
        total_loss = self.lambda_class * class_loss + self.lambda_box * box_loss

        return total_loss