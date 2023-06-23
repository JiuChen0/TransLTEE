import torch

def to_device(data, device):
    """
    Move tensors to the specified device.
    data: Data to be moved. It can be a single tensor or a list or dictionary of tensors.
    device: Target device.
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
    Compute classification accuracy.
    output: Output from the model, shape=[batch_size, num_classes].
    target: Actual labels, shape=[batch_size].
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred == target
        accuracy = correct.float().mean()
        return accuracy.item()
