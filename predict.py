import torch
from PIL import Image
import torchvision.transforms as transforms
from model import TuduiModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
        class_scores = torch.softmax(class_pred, dim=0)
        # class_scores = torch.sigmoid(class_pred)
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

    def predict_random_images(self, image_folder, n_images, output_folder, conf_threshold=0.5):
        """
        从指定文件夹中随机选择N张图片进行预测，并保存结果
        
        Args:
            image_folder (str): 图片所在文件夹路径
            n_images (int): 要预测的图片数量
            output_folder (str): 预测结果保存的文件夹路径
            conf_threshold (float): 置信度阈值
        """
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 如果图片数量不足，调整n_images
        n_images = min(n_images, len(image_files))
        
        # 随机选择图片
        selected_images = random.sample(image_files, n_images)
        
        print(f"将预测 {n_images} 张图片:")
        
        # 对每张图片进行预测
        for i, image_file in enumerate(selected_images, 1):
            image_path = os.path.join(image_folder, image_file)
            save_path = os.path.join(output_folder, f"predict_{i}_{image_file}")
            
            print(f"\n处理第 {i}/{n_images} 张图片: {image_file}")
            
            # 进行预测和可视化
            self.visualize(image_path, conf_threshold, save_path)
            
            # 获取详细的预测结果
            bbox, class_name, confidence = self.predict(image_path, conf_threshold)
            if bbox is not None:
                print(f"检测到物体: {class_name}")
                print(f"置信度: {confidence:.2f}")
                print(f"边界框 (x, y, w, h): {bbox}")
            else:
                print("未检测到置信度足够高的物体")

if __name__ == '__main__':
    # 使用示例
    detector = ObjectDetector(
        model_path='sgd_momentum_best_model.pth',  # 替换为你的模型路径
        num_classes=20
    )
    
    # 预测多张随机图片
    image_folder = "../dataset/VOCdevkit/VOC2007/JPEGImages"  # 图片文件夹路径
    output_folder = "./predict"  # 预测结果保存路径
    n_images = 5  # 要预测的图片数量
    
    detector.predict_random_images(
        image_folder=image_folder,
        n_images=n_images,
        output_folder=output_folder,
        conf_threshold=0.1
    ) 