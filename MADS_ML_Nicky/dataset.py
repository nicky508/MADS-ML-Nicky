import random
import numpy as np
import torch

class BaseDataset:
    def __init__(self) -> None:
        self.process_data()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.targets[idx]

    def process_data(self) -> None:
        # note we raise an error here. This is a template, and we want to force
        # the implementation of this function to be done in the child class
        raise NotImplementedError
    
class Random3dDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        observations = (1000, )
        datasize = (16, 32, 64)
        targetsize = (1,)

        dim = observations + datasize
        dim_target = observations + targetsize

        self.dataset = torch.rand(dim)
        self.targets = torch.randint(0, 2, size=dim_target) 
    
    def process_data(self) -> None:
        pass
