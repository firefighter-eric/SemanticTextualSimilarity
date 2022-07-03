from os.path import join

import pandas as pd


def process(in_path, out_path):
    with open(in_path, encoding='utf8') as fin:
        f = fin.read().split('\n')
    raw = [line.split('\t') for line in f if line]
    data = [[_[5], _[6], float(_[4])] for _ in raw]
    df = pd.DataFrame(data, columns=['s1', 's2', 'score'])
    print(df)
    df.to_csv(out_path, encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    from conf import ROOT

    data_dir = join(ROOT, 'data', 'STS')
    for split in ['train', 'dev', 'test']:
        _in_path = join(data_dir, 'raw', f'sts-{split}.csv')
        _out_path = join(data_dir, 'processed', f'{split}.csv')
        process(_in_path, _out_path)
