import numpy as np
import pandas as pd
import tensorflow as tf

class IHDataset:
    def __init__(self, csv_file):
        """
        Initialize the dataset
        csv_file: Path of the csv file for the dataset
        """
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        """
        Return the size of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the sample at index idx
        idx: Index of the sample
        """
        # The actual implementation might vary depending on the structure of your CSV file
        sample = self.data.iloc[idx]
        features = sample[5:].values  # Extract features from the CSV row
        treatment = sample[0]  # Extract treatment from the CSV row
        outcomes = np.concatenate([sample[1:6], sample[-1]])  # Extract outcomes from the CSV row
        return features, treatment, outcomes

def get_dataset(csv_file):
    """
    Create a tf.data.Dataset
    csv_file: Path of the csv file for the dataset
    """
    ih_dataset = IHDataset(csv_file)
    # Use map to apply __getitem__ to each element in the dataset
    dataset = tf.data.Dataset.from_tensor_slices((range(len(ih_dataset)))).map(
        lambda x: tf.numpy_function(ih_dataset.__getitem__, [x], [tf.float32, tf.float32, tf.float32]))
    return dataset

def get_dataloader(csv_file, batch_size):
    """
    Create a data loader
    csv_file: Path of the csv file for the dataset
    batch_size: Size of each batch
    """
    dataset = get_dataset(csv_file)
    dataloader = dataset.shuffle(len(dataset)).batch(batch_size)
    return dataloader
