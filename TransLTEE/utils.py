#包含各种工具函数，比如模型保存和加载、计算评价指标等。
import torch

def to_device(data, device):
    """
    将张量移动到指定的设备上。
    data: 需要移动的数据，可以是单个张量，也可以是张量的列表或字典。
    device: 目标设备。
    """
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        raise TypeError('Unsupported data type {}'.format(type(data)))

def compute_accuracy(output, target):
    """
    计算分类精度。
    output: 模型的输出，shape=[batch_size, num_classes]。
    target: 真实的标签，shape=[batch_size]。
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred == target
        accuracy = correct.float().mean()
        return accuracy.item()
