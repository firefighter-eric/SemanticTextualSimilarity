from dataclasses import dataclass
from os.path import join

from dataset.datamodule import DataModuleConfig
from task.simcse_task import SimCSETaskConfig
from conf import data_conf, ROOT


@dataclass
class SimCSEScriptConfig(DataModuleConfig, SimCSETaskConfig):
    default_root_dir: str = join(ROOT, 'outputs')
    pass


sts_config = SimCSEScriptConfig(name='sts', data_path=data_conf.sts_path)
