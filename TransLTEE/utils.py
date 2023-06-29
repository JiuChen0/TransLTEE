import tensorflow as tf
import os

def to_device(data, device):
    """
    Move tensors to the specified device.
    data: Data to be moved. It can be a single tensor or a list or dictionary of tensors.
    device: Target device.
    """
    # Use .to() method to move data to device in TensorFlow
    if tf.is_tensor(data):
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
    pred = tf.argmax(output, axis=1)
    correct = tf.equal(pred, target)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy.numpy()

def save_model(model, config, epoch):
    """
    Save the model's state dict and optimizer's state dict.
    model: The model instance.
    config: The config instance.
    epoch: The current epoch number.
    """
    model_dir = os.path.join(config.save_dir, f'model_epoch_{epoch}.pt')
    tf.keras.models.save_model(model, model_dir)
    print(f"Model saved at {model_dir}")

def load_model(model, config, device, epoch):
    """
    Load the model's state dict and optimizer's state dict.
    model: The model instance.
    config: The config instance.
    device: The device instance.
    epoch: The epoch number to load.
    """
    model_dir = os.path.join(config.save_dir, f'model_epoch_{epoch}.pt')
    if os.path.exists(model_dir):
        model = tf.keras.models.load_model(model_dir)
        model = to_device(model, device)
        print(f"Model loaded from {model_dir}")
    else:
        print(f"No model found at {model_dir}, please check the epoch number.")
    return model
