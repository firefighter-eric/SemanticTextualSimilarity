import json
from dataclasses import dataclass
from os.path import join
from pprint import pprint

from conf import ROOT


@dataclass
class TrainerConfig:
    default_root_dir: str = join(ROOT, 'outputs')
    val_check_interval: int = 1000
    monitor: str = 'val/loss'
    mode: str = 'min'


def json_config(cls):
    def inner(_cls):
        _cls.from_json = classmethod(from_json)
        return _cls

    return inner(cls)


def from_json(cls, path):
    with open(path, 'r') as f:
        cls = cls(**json.load(f))
    for k, v in cls.__dict__.items():
        if k.endswith('_path'):
            cls.__dict__[k] = join(ROOT, v)
    pprint(cls.__dict__)
    return cls
