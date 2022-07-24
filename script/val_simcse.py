from os.path import join
from pprint import pprint

from pytorch_lightning import Trainer

from dataset.datamodule import DataModule
from task.simcse_task import SimCSETask
from conf import ROOT

ckpt_path = '../outputs/wiki_unsupervised_bert-base-uncased-v3.ckpt'

task = SimCSETask.load_from_checkpoint(ckpt_path)
config = task.config
# config.model_max_length = 512
pprint(config.__dict__)
dm = DataModule(config)

trainer = Trainer(default_root_dir=join(ROOT, '.tmp'),
                  gpus=1,
                  precision=16,
                  max_epochs=-1)

trainer.validate(task, dm)
