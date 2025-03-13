import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import VOCDataset
from model import TuduiModel
from utils import get_transforms
from loss import DetectionLoss
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        
        # 前向传播
        predictions = model(images)
        
        # 计算损失
        loss = criterion(predictions, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 超参数
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128
    NUM_EPOCHS = 8000
    
    # 数据集路径
    train_img_dir = r"../dataset/VOCdevkit/VOC2007/JPEGImages"
    train_label_dir = r"../dataset/VOCdevkit/VOC2007/Annotations"
    val_img_dir = r"../dataset/VOCdevkit/VOC2007/JPEGImages"
    val_label_dir = r"../dataset/VOCdevkit/VOC2007/Annotations"

    # 创建数据集和数据加载器
    train_dataset = VOCDataset(
        image_folder=train_img_dir,
        label_folder=train_label_dir,
        transform=get_transforms(train=True)
    )
    
    val_dataset = VOCDataset(
        image_folder=val_img_dir,
        label_folder=val_label_dir,
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=32,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=32,
        pin_memory=True
    )
    
    # 创建模型
    model = TuduiModel().to(device)
    
    # 定义损失函数和优化器
    criterion = DetectionLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.1,
    #     patience=5,
    #     verbose=True
    # )
    
    # Create TensorBoard writer
    writer = SummaryWriter('runs/sgd_128_1e4_8000')
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # Add TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 学习率调整
        # scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'sgd_best_model.pth')
            print("Saved best model!")
    
    # Close writer at the end
    writer.close()


if __name__ == "__main__":
    main()
