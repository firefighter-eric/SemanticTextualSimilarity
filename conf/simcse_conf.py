from dataclasses import dataclass

from conf.base_config import TrainerConfig, json_config
from dataset.datamodule import DataModuleConfig
from task.simcse_task import SimCSETaskConfig


@json_config
@dataclass
class SimCSEScriptConfig(DataModuleConfig, SimCSETaskConfig, TrainerConfig):
    pass

# sts_bert_config = SimCSEScriptConfig(name='sts',
#                                      train_data_path=data_conf.sts_path,
#                                      val_data_path=data_conf.sts_path,
#                                      train_batch_size=64,
#                                      lr=3e-5,
#                                      mlm_flag=False)

# wiki_bert_config = SimCSEScriptConfig(name='sts',
#                                       train_data_path=data_conf.wiki_path,
#                                       val_data_path=data_conf.sts_path,
#                                       train_batch_size=64,
#                                       lr=3e-5,
#                                       mlm_flag=False,
#                                       model_max_length=64)
