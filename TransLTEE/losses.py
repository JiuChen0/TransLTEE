#包含所有的损失函数定义,Wasserstein-1 distance等

import torch

def compute_train_loss(model, batch, criterion):
    """
    计算训练损失
    model: 模型对象
    batch: 一批训练数据
    criterion: 损失函数
    """
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return loss

def compute_valid_loss(model, batch, criterion):
    """
    计算验证损失
    model: 模型对象
    batch: 一批验证数据
    criterion: 损失函数
    """
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return loss
######################
#如果模型返回多个输出，或者损失函数需要额外的参数，可能需要在这些函数中进行相应的调整
######################