import pandas as pd
from torch.utils.data import Dataset, DataLoader

class IHDataset(Dataset):
    def __init__(self, csv_file):
        """
        Initialize the dataset
        csv_file: Path of the csv file for the dataset
        """
        self.data = pd.read_csv(csv_file)

    def __getitem__(self, idx):
        """
        Get the sample at index idx
        idx: Index of the sample
        """
        return self.data.iloc[idx]

    def __len__(self):
        """
        Return the size of the dataset
        """
        return len(self.data)

def get_dataloader(csv_file, batch_size):
    """
    Create a data loader
    csv_file: Path of the csv file for the dataset
    batch_size: Size of each batch
    """
    dataset = IHDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
