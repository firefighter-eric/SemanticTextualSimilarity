from os.path import join
from matplotlib import pyplot as plt

from conf import ROOT

with open(join(ROOT, 'data', 'wiki', 'wiki1m.txt'), 'r') as f:
    d = f.read().strip().split('\n')

print(f'{len(d)=}')

length = list(map(len, d))

plt.hist(length, bins=100, range=(0, 512))
plt.show()
