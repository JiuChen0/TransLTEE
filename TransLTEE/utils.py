import tensorflow as tf

def to_device(data, device):
    """
    Move tensors to the specified device.
    data: Data to be moved. It can be a single tensor or a list or dictionary of tensors.
    device: Target device.
    """
    if isinstance(data, tf.Tensor):
        return data.device(device)
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
    pred = tf.argmax(output, axis=1)
    correct = tf.equal(pred, target)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy.numpy()
