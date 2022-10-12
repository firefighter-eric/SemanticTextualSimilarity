from dataclasses import dataclass
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset import STSSupervisedDataset
from dataset import data_augment
from dataset.sts_dataset import OneSentDataset
from model.tokenizer import Tokenizer, TokenizerConfig


@dataclass
class DataModuleConfig(TokenizerConfig):
    name: str = ''
    train_data_path: str = ''
    val_data_path: str = ''
    train_batch_size: int = 64
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
        sent = []
        for i in range(len(s1)):
            sent.append(s1[i])
            sent.append(s2[i])

        sent = self.tokenizer(sent)
        if self.config.mlm_flag:
            sent, mlm_label = data_augment.get_mlm_label(s)
        else:
            mlm_label = None
        score = torch.FloatTensor(score)

        return {'sent': sent,
                'mlm_label': mlm_label,
                'label': score}

    def setup(self, stage: Optional[str] = None):
        # self.train_data = STSUnsupervisedDataset(path=self.config.train_data_path, split='train')
        self.train_data = OneSentDataset(path=self.config.train_data_path)
        self.val_data = STSSupervisedDataset(path=self.config.val_data_path)

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
