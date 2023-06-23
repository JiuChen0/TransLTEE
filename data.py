import pandas as pd
from torch.utils.data import Dataset, DataLoader

class IHDataset(Dataset):
    def __init__(self, csv_file):
        """
        初始化数据集
        csv_file: 数据集的csv文件路径
        """
        self.data = pd.read_csv(csv_file)

    def __getitem__(self, idx):
        """
        获取索引为idx的样本
        idx: 样本的索引
        """
        return self.data.iloc[idx]

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)

def get_dataloader(csv_file, batch_size):
    """
    创建数据加载器
    csv_file: 数据集的csv文件路径
    batch_size: 每个批次的大小
    """
    dataset = IHDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
