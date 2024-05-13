import torch
import numpy as np
from dataset import Random3dDataset
from mads_datasets.base import BaseDatastreamer

## 1_pytorch_intro assignment 1 and 2

#Create Random3DDataset which is abstracted from BaseDataset and contains 1000 Random 3d Tensors (16, 32, 64), with a random label
dataset = Random3dDataset()

#Gives 3D tensor indexed zero en print the amount of 3D tensors in the dataset
# X, y = dataset[0]
# print(len(dataset))
# print(X.shape, y.shape)

# Batch processor for "untupling" te tensor/label tensors
def batch_processor(batch):
    X, Y = zip(*batch)
    return np.stack(X), np.array(Y)

#used the BaseDatastreamer with the Random3DDataset, creating batches of size 16 with the batch_processor
streamer = BaseDatastreamer(
    dataset=dataset,
    batchsize=16,
    preprocessor=batch_processor
)

#Generates batches of size 16 on every run.
gen = streamer.stream()
X, y = next(gen)
print(X.shape, y.shape)

