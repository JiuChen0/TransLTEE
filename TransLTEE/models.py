#包含了所有的模型，如DoubleHeadRNN、TransformerEncoder、MyModel
import torch
from torch import nn
from transformers import TransformerModel
from losses import WassersteinLoss

class DoubleHeadRNN(nn.Module):
    """ 
    双头RNN结构
    """
    def __init__(self, input_dim, hidden_dim):
        """
        构造函数
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        """
        super(DoubleHeadRNN, self).__init__()
        # 定义两个RNN结构
        self.rnn0 = nn.RNN(input_dim, hidden_dim)
        self.rnn1 = nn.RNN(input_dim, hidden_dim)
    
    def forward(self, x):
        """
        前向传播
        x: 输入数据
        """
        # 对输入数据进行处理，得到两个RNN的输出
        output0, _ = self.rnn0(x)
        output1, _ = self.rnn1(x)
        return output0, output1

class TransformerEncoder(nn.Module):
    """
    Transformer编码器结构
    """
    def __init__(self, hidden_dim):
        """
        构造函数
        hidden_dim: 隐藏层维度
        """
        super(TransformerEncoder, self).__init__()
        # 使用预训练的bert作为Transformer编码器
        self.transformer = TransformerModel.from_pretrained('bert-base-uncased')
    
    def forward(self, x):
        """
        前向传播
        x: 输入数据
        """
        # 对输入数据进行处理，得到Transformer编码器的输出
        outputs = self.transformer(x)
        return outputs.last_hidden_state

class MyModel(nn.Module):
    """
    我们的模型结构,包含双头RNN和Transformer编码器
    """
    def __init__(self, input_dim, hidden_dim):
        """
        构造函数
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        """
        super(MyModel, self).__init__()
        # 定义双头RNN
        self.double_head_rnn = DoubleHeadRNN(input_dim, hidden_dim)
        # 定义Transformer编码器
        self.transformer_encoder = TransformerEncoder(hidden_dim)
        # 定义损失函数，这里使用Wasserstein距离
        self.criterion = WassersteinLoss()
    
    def forward(self, x):
        """
        前向传播
        x: 输入数据
        """
        # 获得双头RNN的输出
        output0, output1 = self.double_head_rnn(x)
        # 获得Transformer编码器的输出
        encoded0 = self.transformer_encoder(output0)
        encoded1 = self.transformer_encoder(output1)
        # 计算两个输出之间的Wasserstein距离，作为模型的损失
        loss = self.criterion(encoded0, encoded1)
        return loss


###################
#没有实现注意力机制，以及将多个损失函数叠加的部分
###################