import numpy as np
import os
from scipy.misc import imread, imresize

d = '../datasets/img_align_celeba/'
files = [s for s in os.listdir(d) if s.endswith('.jpg')]
files.sort()

n = min(10000, len(files))
files = files[:n]

a = np.zeros((n, 64, 64, 3), dtype=np.uint8)

for i, fn in enumerate(files):
    a[i] = imresize(imread(d + fn, mode='RGB'), (64, 64))
    print(i)

np.save('../datasets/celeba.npy', a)
