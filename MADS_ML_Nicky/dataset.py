import random
import numpy as np
import torch

class BaseDataset:
    def __init__(self) -> None:
        self.process_data()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def process_data(self) -> None:
        # note we raise an error here. This is a template, and we want to force
        # the implementation of this function to be done in the child class
        raise NotImplementedError
    
class Random3dDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        observations = (50, )
        datasize = (3, 28, 28)

        dim = observations + datasize

        self.data = torch.rand(dim)
        self.targets = torch.randint(0, 2, size=(1,1)) #torch.bernoulli(torch.rand(1))
    
    def process_data(self) -> None:
        pass
