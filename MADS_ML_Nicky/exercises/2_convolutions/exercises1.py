from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger
import warnings
warnings.simplefilter("ignore", UserWarning)

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor
from mltrainer import metrics

from torchsummary import summary
import torch.optim as optim
from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes
from datetime import datetime

fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
batchsize = 64
preprocessor = BasePreprocessor()
streamers = fashionfactory.create_datastreamer(batchsize=batchsize, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

x, y = next(iter(trainstreamer))
print(x.shape, y.shape)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

optimizer = optim.Adam
loss_fn = torch.nn.CrossEntropyLoss()
accuracy = metrics.Accuracy()

# Define the hyperparameter search space
settings = TrainerSettings(
    epochs=5,
    metrics=[accuracy],
    logdir="modellogs",
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD],
)

# Define model
class CNN(nn.Module):
    def __init__(self, filters, units1, units2, input_size=(64, 1, 28, 28)):
        super().__init__()
        
        #Convolution layers
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        #AVG pooling for fully connected layers
        activation_map_size = self._conv_test(input_size)
        logger.info(f"Aggregating activationmap with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        #Full connected layers
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(units2, 10)
        )

    def _conv_test(self, input_size = (64, 1, 28, 28)):
        x = torch.ones(input_size)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits

model = CNN(filters=64, units1=128, units2=64).to(device)
# summary(model, input_size=(1, 28, 28))

trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
trainer.loop()