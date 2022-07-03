from dataclasses import dataclass
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset import STSDataset
from dataset import data_augment
from model.tokenizer import Tokenizer, TokenizerConfig


@dataclass
class DataModuleConfig(TokenizerConfig):
    data_path: str = ''
    batch_size: int = 32
    num_workers: int = 0


class DataModule(LightningDataModule):
    def __init__(self, config: DataModuleConfig):
        super().__init__()
        self.config = config
        self.tokenizer = Tokenizer(config)
        self.train_data = None
        self.val_data = None

    def collate_fn(self, batch):
        s1, s2, _ = zip(*batch)
        s1, s2 = list(s1), list(s2)
        s1 = self.tokenizer(s1)
        s2 = self.tokenizer(s2)
        s1, mlm_label1 = data_augment.get_mlm_label(s1)
        s2, mlm_label2 = data_augment.get_mlm_label(s2)
        return s1, s2, mlm_label1, mlm_label2

    def setup(self, stage: Optional[str] = None):
        self.train_data = STSDataset(path=self.config.data_path, split='train')
        self.val_data = STSDataset(path=self.config.data_path, split='dev')

    def pin_dataloader(self, data):
        return DataLoader(dataset=data, batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn, shuffle=True,
                          drop_last=True, persistent_workers=True, pin_memory=True, prefetch_factor=2)

    def dataloader(self, data, shuffle):
        drop_last = shuffle
        return DataLoader(dataset=data, batch_size=self.config.batch_size, collate_fn=self.collate_fn, shuffle=shuffle,
                          drop_last=drop_last)

    def train_dataloader(self):
        return self.dataloader(self.train_data, True)

    def val_dataloader(self):
        return self.dataloader(self.val_data, False)


if __name__ == '__main__':
    from conf import data_conf

    _config = DataModuleConfig(data_path=data_conf.sts_path)
    _dm = DataModule(_config)
    _dm.setup()

    dl = _dm.val_dataloader()
    sample = next(iter(dl))
