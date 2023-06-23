#文件包含模型的训练代码
import torch
from utils import to_device, compute_accuracy
from losses import compute_train_loss, compute_valid_loss  # 确保在losses.py中定义了这个函数

def train(model, train_dataloader, valid_dataloader, criterion, optimizer, config):
    """
    训练模型
    model: 模型对象
    train_dataloader: 训练数据加载器
    valid_dataloader: 验证数据加载器
    criterion: 损失函数
    optimizer: 优化器
    config: 配置对象
    """

    # 移动模型到设备
    device = config.device
    model.to(device)

    for epoch in range(config.num_epochs):
        # 设置模型为训练模式
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            # 移动数据到设备
            batch = to_device(batch, device)

            # 计算损失
            loss = compute_train_loss(model, batch, criterion)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # 在验证集上评估模型
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_dataloader:
                # 移动数据到设备
                batch = to_device(batch, device)

                # 计算损失
                loss = compute_valid_loss(model, batch, criterion)

                valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)

        # 打印训练和验证损失
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}')

    return model
###################
#需要根据实际需求修改compute_train_loss和compute_valid_loss函数。
###################