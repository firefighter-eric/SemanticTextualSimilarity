from dataclasses import dataclass
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset import STSUnsupervisedDataset, STSSupervisedDataset
from dataset import data_augment
from dataset.sts_dataset import STSOneSentDataset
from model.tokenizer import Tokenizer, TokenizerConfig


@dataclass
class DataModuleConfig(TokenizerConfig):
    name: str = ''
    train_data_path: str = ''
    val_data_path: str = ''
    train_batch_size: int = 128
    val_batch_size: int = 16
    num_workers: int = 0
    mlm_flag: bool = False


class DataModule(LightningDataModule):
    def __init__(self, config: DataModuleConfig):
        super().__init__()
        self.config = config
        self.tokenizer = Tokenizer(config)
        self.train_data = None
        self.val_data = None

    def collate_fn(self, batch):
        s1, s2, score = zip(*batch)
        s1, s2 = list(s1), list(s2)
        s1 = self.tokenizer(s1)
        s2 = self.tokenizer(s2)
        if self.config.mlm_flag:
            s1, mlm_label_1 = data_augment.get_mlm_label(s1)
            s2, mlm_label_2 = data_augment.get_mlm_label(s2)
        else:
            mlm_label_1, mlm_label_2 = None, None
        score = torch.FloatTensor(score)

        return {'s1': s1,
                's2': s2,
                'mlm_label_1': mlm_label_1,
                'mlm_label_2': mlm_label_2,
                'label': score}

    def setup(self, stage: Optional[str] = None):
        # self.train_data = STSUnsupervisedDataset(path=self.config.train_data_path, split='train')
        self.train_data = STSOneSentDataset(path=self.config.train_data_path)
        self.val_data = STSSupervisedDataset(path=self.config.val_data_path, split='dev')

    def pin_dataloader(self, data):
        return DataLoader(dataset=data, batch_size=self.config.train_batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn, shuffle=True,
                          drop_last=True, persistent_workers=True, pin_memory=True, prefetch_factor=2)

    def dataloader(self, data, batch_size, shuffle):
        drop_last = shuffle
        return DataLoader(dataset=data, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle,
                          drop_last=drop_last)

    def train_dataloader(self):
        return self.dataloader(self.train_data, batch_size=self.config.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.val_data, batch_size=self.config.val_batch_size, shuffle=False)


if __name__ == '__main__':
    from conf import data_conf

    _config = DataModuleConfig(train_data_path=data_conf.wiki_path,
                               val_data_path=data_conf.sts_path,
                               model_max_length=64)
    _dm = DataModule(_config)
    _dm.setup()

    dl = _dm.train_dataloader()
    sample = next(iter(dl))
