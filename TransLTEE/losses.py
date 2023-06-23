#包含所有的损失函数定义,Wasserstein-1 distance等

import torch
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F

class WassersteinLoss(Module):
    """
    Wasserstein-1 distance
    """
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        计算两个向量的Wasserstein-1 distance
        x, y: 输入的向量
        返回值: Wasserstein-1 distance
        """
        return torch.mean(x-y)**2  # 这是简化版本的Wasserstein距离，仅用于示例，真正的Wasserstein距离计算可能需要更复杂的操作
