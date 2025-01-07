import torch
from torch import nn


class TuduiModel(nn.Module):
    def __init__(self, num_classes=20):
        super(TuduiModel, self).__init__()
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 全连接层将特征映射到输出
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 52 * 52, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes + 4)  # num_classes个类别 + 4个边界框坐标
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        # 分离类别预测和边界框预测
        class_pred = x[:, :self.num_classes]  # 类别预测
        # todo: drop???
        box_pred = torch.sigmoid(x[:, self.num_classes:])  # 边界框预测 (归一化到0-1)

        # 组合预测结果
        output = torch.cat([class_pred, box_pred], dim=1)
        return output


if __name__ == '__main__':
    # 测试代码
    input = torch.randn(1, 3, 418, 418)
    model = TuduiModel(16)
    output = model(input)
    print("Output shape:", output.shape)
    print("Sample output:", output[0])
