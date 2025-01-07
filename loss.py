import torch


class DetectionLoss(torch.nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()  # 改用交叉熵损失
        self.num_classes = num_classes
        # 损失权重
        self.lambda_class = 1.0  # 类别损失权重
        self.lambda_box = 1.0  # 边界框坐标损失权重
        self.lambda_iou = 1.0  # IOU损失权重

    def forward(self, predictions, targets):
        """
        计算目标检测的损失
        predictions: (batch_size, num_classes + 4) -> [class_scores, x, y, w, h]
        targets: (batch_size, 5) -> [class_id, x, y, w, h]
        """
        # 1. 类别损失 - 使用交叉熵
        class_pred = predictions[:, :self.num_classes]  # 类别预测分数
        class_target = targets[:, 0].long()  # 目标类别ID
        class_loss = self.ce(class_pred, class_target)

        # 2&3. 边界框损失和IOU损失
        mask = targets[:, 0] >= 0  # 有效标注的mask
        if mask.sum() > 0:
            # 边界框坐标损失（MSE）
            box_pred = predictions[mask, self.num_classes:]  # 预测的边界框坐标
            box_target = targets[mask, 1:]  # 目标边界框坐标
            box_loss = self.mse(box_pred, box_target)

            # IOU损失
            iou = intersection_over_union(box_pred, box_target)
            iou_loss = 1 - iou.mean()

            # 总损失 - 加权组合
            loss = (self.lambda_class * class_loss +
                    self.lambda_box * box_loss +
                    self.lambda_iou * iou_loss)
        else:
            loss = self.lambda_class * class_loss

        return loss