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
        return self.data.iloc[idx]


def get_dataset(csv_file):
    """
    Create a tf.data.Dataset
    csv_file: Path of the csv file for the dataset
    """
    ih_dataset = IHDataset(csv_file)
    dataset = tf.data.Dataset.from_tensor_slices(ih_dataset.data.values)
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
