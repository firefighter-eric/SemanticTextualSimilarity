from dataclasses import dataclass
from os.path import join

from dataset.datamodule import DataModuleConfig
from task.simcse_task import SimCSETaskConfig
from conf import data_conf, ROOT


@dataclass
class SimCSEScriptConfig(DataModuleConfig, SimCSETaskConfig):
    default_root_dir: str = join(ROOT, 'outputs')
    val_check_interval: int = 1000,
    pass


sts_bert_config = SimCSEScriptConfig(name='sts',
                                     train_data_path=data_conf.sts_path,
                                     val_data_path=data_conf.sts_path,
                                     train_batch_size=64,
                                     lr=3e-5,
                                     mlm_flag=False)

wiki_bert_config = SimCSEScriptConfig(name='sts',
                                      train_data_path=data_conf.wiki_path,
                                      val_data_path=data_conf.sts_path,
                                      train_batch_size=64,
                                      lr=3e-5,
                                      mlm_flag=False,
                                      model_max_length=64)
