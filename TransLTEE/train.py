#文件包含模型的训练代码
import torch
from models import MyModel
from config import Config

def train(model, dataloader, criterion, optimizer, config):
    """
    训练函数
    model: 模型
    dataloader: 数据加载器
    criterion: 损失函数
    optimizer: 优化器
    config: 配置参数
    """
    model.train()  # 将模型设置为训练模式
    for epoch in range(config.epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            # 迁移到GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练信息
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')

if __name__ == '__main__':
    config = Config()

    # 创建模型
    model = MyModel(config.input_dim, config.hidden_dim).to(device)

    # 创建数据加载器，这里假设你已经有了一个可以用的DataLoader
    # dataloader = DataLoader(dataset, batch_size=config.batch_size)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练模型
    train(model, dataloader, criterion, optimizer, config)
