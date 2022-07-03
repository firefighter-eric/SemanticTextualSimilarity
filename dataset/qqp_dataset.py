import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class QQPDataset(Dataset):
    def __init__(self, path, split='train'):
        df = pd.read_csv(path)
        data = df[["question1", "question2", "is_duplicate"]].values.tolist()
        train, dev = train_test_split(data, test_size=0.1, random_state=43)
        self.data = train if split == 'train' else dev

    def __getitem__(self, index):
        """
        [s1: str, s2: str, label: int]
        """
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    _d = QQPDataset()
