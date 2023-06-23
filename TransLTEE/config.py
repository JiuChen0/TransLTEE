#包含项目的所有配置，如学习率、批次大小、训练轮次、模型参数等
class Config:
    """
    模型和训练过程中的配置
    """
    def __init__(self):
        # 训练参数
        self.epochs = 100  # 训练轮次
        self.batch_size = 64  # 批次大小
        self.learning_rate = 0.001  # 学习率
        self.weight_decay = 0.0001  # 权重衰减
        self.save_dir = './home'  # 模型保存路径

        # 模型参数
        self.input_dim = 128  # 输入维度
        self.hidden_dim = 256  # 隐藏层维度
