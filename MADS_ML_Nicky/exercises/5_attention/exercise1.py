from pathlib import Path
from EegDataset import EegDataset
from pathlib import Path
import gin
from DatasetFactory import DatasetFactoryProvider
from typing import List
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import mlflow
from torch.nn.utils.rnn import pad_sequence
from mltrainer import rnn_models, Trainer
from torch import optim
import pathlib
from mltrainer import TrainerSettings, ReportTypes
from mltrainer.metrics import Metric
from typing import Dict

ginModelPath = Path(__file__).parent.resolve() / "eeg_model.gin"

class Accuracy(Metric):
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        yhat = (yhat > 0.5).astype(int)
        return (yhat == y).sum() / len(yhat)
    
@gin.configurable
class EegModel(nn.Module):
    def __init__(
        self,
        config: Dict
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config["input_size"], config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], 1),
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yhat = self.fc(x)
        return yhat.squeeze(1)
    
def getData():
    eegdatasetfactory = DatasetFactoryProvider.create_factory('EEG')
    streamers = eegdatasetfactory.create_datastreamer(batchsize=32)
    train = streamers["train"]
    valid = streamers["valid"]

    trainstreamer = train.stream()
    validstreamer = valid.stream()

    return train, valid, trainstreamer, validstreamer
    
def getSettings(train, valid):
    accuracy = Accuracy()
    settings = TrainerSettings(
        epochs=10,
        metrics=[accuracy],
        logdir=Path("eeg"),
        train_steps=len(train),
        valid_steps=len(valid),
        optimizer_kwargs = {"lr": 1e-2},
        reporttypes=[ReportTypes.GIN, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs={"patience": 5},
    )    
    return settings

def getTrainer(settings, trainstreamer, validstreamer, model):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.BCELoss()
    trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optim.Adam,
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau
        )
    return trainer

def trainModel(trainer, model):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("eeg")
    modeldir = Path("../../models/eeg/").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)

    with mlflow.start_run():
        mlflow.set_tag("model", "Eegmodel")
        mlflow.set_tag("dev", "nicky")
        mlflow.log_params(gin.get_bindings("EegModel"))

        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        torch.save(model, modelpath)
        trainer.loop()
        
def main():
    train, valid, trainstreamer, validstreamer = getData()
    settings = getSettings(train, valid)
    gin.parse_config_file(ginModelPath)
    model = EegModel()
    print(model)
    trainer = getTrainer(settings, trainstreamer, validstreamer, model)
    
    trainModel(trainer, model)

    
if __name__ == "__main__":
    main()