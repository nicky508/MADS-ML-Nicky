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

#With batchsize 4 almost no learning is happening. The loss lines are almost straight app. 0.54
#With batchsize 32 results are quite as good as with 64 app. 0.3
#Batchsize 128 performs almost the same as 64 app. 0.37
#with batchsize 512 the results are slightly worse. app. 0.43 
streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

print(gin.config_str())

accuracy = metrics.Accuracy()

gin.parse_config_file(ginModelPath)

# units = [64, 32, 16] makes it less accurate (app. 0.54 instead of app. 0.34) but spread more widely over the Y axis. 
# units = [1024, 512, 256] Makes it not more accurate, about same as [256, 128, 64] but takes much longer without increasing accuracy
units = [512, 256, 128, 64]
loss_fn = torch.nn.CrossEntropyLoss()

settings = TrainerSettings(
    epochs=5,
    metrics=[accuracy],
    logdir="modellogs",
    train_steps=len(train),
    valid_steps=len(valid),
    optimizer_kwargs = {"lr": 1e-3},
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.GIN],
)

for unit1 in units:
    for unit2 in units:
        gin.bind_parameter("NeuralNetwork.units1", unit1)
        gin.bind_parameter("NeuralNetwork.units2", unit2)

#SGD Stochastic gradient descent optimizer performs worse app. 0.5

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
