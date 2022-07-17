from pprint import pprint

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from conf import simcse_conf
from dataset.datamodule import DataModule
from task.simcse_task import SimCSETask

config = simcse_conf.wiki_bert_config
pprint(config.__dict__)

dm = DataModule(config)
task = SimCSETask(config, from_pretrained=True)

model_name = f'{config.name}_{config.model_name}'
logger = TensorBoardLogger(save_dir=config.default_root_dir, name=model_name)
checkpoint_callback = ModelCheckpoint(dirpath=config.default_root_dir, filename=model_name,
                                      monitor='val/loss', save_top_k=1, save_weights_only=True, mode="min")
early_stopping = EarlyStopping(monitor='val/loss', patience=3, mode="min")

trainer = Trainer(logger=logger,
                  callbacks=[checkpoint_callback, early_stopping],
                  default_root_dir=config.default_root_dir,
                  val_check_interval=1000,
                  gpus=1,
                  precision=16,
                  max_epochs=100)

trainer.validate(task, dm)
trainer.fit(task, dm)
