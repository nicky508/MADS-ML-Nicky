from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, default_collate

class EegDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.dataset: list = []
        self.process_data()

    def __len__(self):
        # this should return the size of the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        return self.dataset[idx]
    
    def process_data(self):
        for obs in self.data:
            ar = np.array(obs.tolist())
            x = ar[:14].astype(float)
            y = ar[14].astype(int)
            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor([y], dtype=torch.float32)  # Ensure y has shape [1]

            self.dataset.append((x, y))