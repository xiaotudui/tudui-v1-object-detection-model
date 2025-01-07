import torch
from PIL import Image
import torchvision.transforms as transforms
from model import TuduiModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDetector:
    def __init__(self, model_path, num_classes=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TuduiModel(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((418, 418)),
            transforms.ToTensor(),
        ])
        
        # VOC数据集的类别名称（按需修改）
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def predict(self, image_path, conf_threshold=0.5):
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # 分离类别预测和边界框预测
        class_pred = predictions[0, :20]
        box_pred = predictions[0, 20:]
        
        # 获取最高置信度的类别
        class_scores = torch.sigmoid(class_pred)
        max_score, predicted_class = torch.max(class_scores, 0)
        
        if max_score < conf_threshold:
            return None, None, None
        
        # 转换边界框坐标（相对坐标转绝对坐标）
        x, y, w, h = box_pred.cpu().numpy()
        x = x * orig_size[0]
        y = y * orig_size[1]
        w = w * orig_size[0]
        h = h * orig_size[1]
        
        return (x, y, w, h), self.class_names[predicted_class], max_score.item()

    def visualize(self, image_path, conf_threshold=0.5, save_path=None, figsize=(10, 10)):
        # 加载原始图像
        image = Image.open(image_path).convert('RGB')
        
        # 获取预测结果
        bbox, class_name, confidence = self.predict(image_path, conf_threshold)
        
        if bbox is None:
            print("No object detected with confidence above threshold.")
            return
        
        # 创建图形，设置图像大小
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)
        
        # 绘制边界框
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x - w/2, y - h/2), w, h,
            linewidth=2, edgecolor='r', facecolor='none',
            label=f'{class_name}'
        )
        ax.add_patch(rect)
        
        # 添加类别标签和置信度，改进文本显示
        label = f'{class_name}: {confidence:.2f}'
        plt.text(
            x - w/2, y - h/2 - 10,
            label,
            color='white',
            fontsize=12,
            bbox=dict(
                facecolor='red',
                alpha=0.8,
                pad=3,
                edgecolor='none'
            )
        )
        
        # 添加图例
        plt.legend(loc='upper right')
        
        # 设置标题
        plt.title(f'Object Detection Result\nConfidence Threshold: {conf_threshold}')
        
        # 移除坐标轴
        plt.axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像如果指定了保存路径
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()


if __name__ == '__main__':
    # 使用示例
    detector = ObjectDetector(
        model_path='best_model.pth',  # 替换为你的模型路径
        num_classes=20
    )
    
    # 预测单张图片
    image_path = r"D:\xiaotudui\Dataset\VOC-2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000200.jpg"  # 替换为你的图片路径
    detector.visualize(image_path, conf_threshold=0.1)
    
    # 如果只需要获取预测结果而不需要可视化
    bbox, class_name, confidence = detector.predict(image_path)
    if bbox is not None:
        print(f"检测到物体: {class_name}")
        print(f"置信度: {confidence:.2f}")
        print(f"边界框 (x, y, w, h): {bbox}") 