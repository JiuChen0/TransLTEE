#作为项目的入口，调用上述各个文件中的函数来完成项目的整个流程
import torch
from models import MyModel
from config import Config
from train import train
from data import get_dataloader  # 导入data.py中定义的函数

def main():
    # 创建配置对象
    config = Config()

    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型，并移动到GPU
    model = MyModel(config.input_dim, config.hidden_dim).to(device)

    # 创建数据加载器
    train_dataloader = get_dataloader('train.csv', config.batch_size)
    valid_dataloader = get_dataloader('valid.csv', config.batch_size)  # 如果有验证集的话

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 训练模型
    train(model, train_dataloader, valid_dataloader, criterion, optimizer, config)  # 这里的train函数需要修改，以接受验证集的数据加载器

    # 保存模型
    torch.save(model.state_dict(), f'{config.save_dir}/model.pt')

if __name__ == '__main__':
    main()
