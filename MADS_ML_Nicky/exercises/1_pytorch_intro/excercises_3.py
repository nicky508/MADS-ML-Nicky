from mads_datasets import DatasetFactoryProvider, DatasetType

from mltrainer.preprocessors import BasePreprocessor
from mltrainer import imagemodels, Trainer, TrainerSettings, ReportTypes, metrics

import torch.optim as optim
import gin
import torch
import os 

ginModelPath = os.path.join(os.path.dirname(__file__), "model.gin")
gin.parse_config_file(ginModelPath)

preprocessor = BasePreprocessor()
fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

print(gin.config_str())

accuracy = metrics.Accuracy()

gin.parse_config_file(ginModelPath)

units = [256, 128, 64]
loss_fn = torch.nn.CrossEntropyLoss()

settings = TrainerSettings(
    epochs=5,
    metrics=[accuracy],
    logdir="modellogs",
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.GIN],
)

for unit1 in units:
    for unit2 in units:
        gin.bind_parameter("NeuralNetwork.units1", unit1)
        gin.bind_parameter("NeuralNetwork.units2", unit2)

        model = imagemodels.NeuralNetwork()
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optim.Adam,
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau
        )
        trainer.loop()
