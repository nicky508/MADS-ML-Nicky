import torch
from dataset import Random3dDataset

dataset = Random3dDataset()
X, y = dataset[0]
print(X.shape, y)
