from os.path import join

import pandas as pd
from torch.utils.data import Dataset


class OneSentDataset(Dataset):
    def __init__(self, path, split='train'):
        with open(path, 'r') as f:
            data = f.read().strip().split('\n')
        # train, dev = train_test_split(data, test_size=0.01, random_state=42)
        # self.data = train if split == 'train' else dev
        self.data = data

    def __getitem__(self, index):
        """
        [s1: str, s2: str, label: int]
        """
        sample = self.data[index]
        return [sample, sample, 1.0]

    def __len__(self):
        return len(self.data)


class STSUnsupervisedDataset(Dataset):
    def __init__(self, path, split):
        df = pd.read_csv(join(path, f'{split}.csv'))
        self.data = df['s1'].to_list() + df['s2'].to_list()

    def __getitem__(self, index):
        """
        [s1: str, s2: str, label: int]
        """
        s = self.data[index]
        return [s, s, 1.0]

    def __len__(self):
        return len(self.data)


class STSSupervisedDataset:
    def __init__(self, path, split):
        df = pd.read_csv(join(path, f'{split}.csv'))
        self.data = df[['s1', 's2', 'score']].values.tolist()

    def __getitem__(self, index):
        """
        [s1: str, s2: str, label: int]
        """
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from conf import data_conf

    _d = STSUnsupervisedDataset(data_conf.sts_path, split='train')
