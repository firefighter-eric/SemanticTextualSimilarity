from os.path import join
from pprint import pprint

from pytorch_lightning import Trainer

from dataset.datamodule import DataModule
from task.simcse_task import SimCSETask
from conf import ROOT

ckpt_path = 'C:\Projects\SemanticTextualSimilarity\outputs\sts_bert-base-cased-v1.ckpt'

task = SimCSETask.load_from_checkpoint(ckpt_path)
config = task.config
pprint(config.__dict__)
dm = DataModule(config)

trainer = Trainer(default_root_dir=join(ROOT, '.tmp'),
                  gpus=1,
                  precision=16,
                  max_epochs=-1)

trainer.validate(task, dm)
