from pathlib import Path
from mads_datasets import datatools
from typing import Any, Mapping
from scipy.io import arff
import numpy as np
from EegDataset import EegDataset
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)
from mltrainer.preprocessors import PaddedPreprocessor


class DatasetFactoryProvider:
    @staticmethod
    def create_factory(dataset_type, **kwargs):
        if dataset_type == 'EEG':
            return EegDatasetFactory()

        raise ValueError(f"Invalid dataset type: {dataset_type}")


class EegDatasetFactory():
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ):
        data_dir = Path(kwargs.get("datadir", Path.home() / ".cache/mads_datasets"))
        filename = "EGG.arff"
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        datatools.get_file(data_dir=data_dir, filename=filename, url=url, unzip=False)
        datapath = data_dir / filename

        data = arff.loadarff(datapath)
        obs = data[0]
    
        trainidx = int(len(obs) * 0.8)
        train = obs[:trainidx]
        valid = obs[trainidx:]
        traindataset = EegDataset(train)
        validdataset = EegDataset(valid)
        return {"train": traindataset, "valid": validdataset}
    
    def create_datastreamer(
        self, batchsize: int
    ):
        datasets = self.create_dataset()
        traindataset = datasets["train"]
        validdataset = datasets["valid"]
        
        preprocessor = PaddedPreprocessor()

        trainstreamer = BaseDatastreamer(
            traindataset, batchsize, preprocessor
        )
        validstreamer = BaseDatastreamer(
            validdataset, batchsize, preprocessor
        )
        return {"train": trainstreamer, "valid": validstreamer}
        
class BaseDatastreamer:
    def __init__(
        self,
        dataset,
        batchsize: int,
        preprocessor
    ) -> None:
        self.dataset = dataset
        self.batchsize = batchsize

        if preprocessor is None:
            self.preprocessor = lambda x: zip(*x)
        else:
            self.preprocessor = preprocessor

        self.size = len(self.dataset)
        self.reset_index()

    def __len__(self) -> int:
        return int(len(self.dataset) / self.batchsize)

    def __repr__(self) -> str:
        return f"BasetDatastreamer: {self.dataset} (streamerlen {len(self)})"

    def reset_index(self) -> None:
        self.index_list = np.random.permutation(self.size)
        self.index = 0

    def batchloop(self) -> Sequence[Tuple]:
        batch = []
        for _ in range(self.batchsize):
            x, y = self.dataset[int(self.index_list[self.index])]
            batch.append((x, y))
            self.index += 1
        return batch

    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            X, Y = self.preprocessor(batch) 
            yield X, Y