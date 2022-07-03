from os.path import join

import pandas as pd
from torch.utils.data import Dataset


class STSDataset(Dataset):
    def __init__(self, path, split):
        df = pd.read_csv(join(path, f'{split}.csv'))
        self.data = df[["s1", "s2", "score"]].values.tolist()

    def __getitem__(self, index):
        """
        [s1: str, s2: str, label: int]
        """
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from conf import data_conf

    _d = STSDataset(data_conf.sts_path, split='train')
