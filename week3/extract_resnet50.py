# extract resnet50 features using multiprocessing

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Flatten
from keras.models import Model
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import numpy as np
from scipy.misc import imread, imresize
import os


batch_size = 64

# resnet50 + flatten

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = Flatten()(resnet.output)
model = Model(inputs=resnet.input, outputs=x)
model.compile('adam', 'mse')


# from - to dirs

in_path = '../datasets/coco/val2017/'
out_path = 'data/resnet50_vectors_val/'

if not os.path.isdir(out_path):
    os.makedirs(out_path)

# filter images

fns = [fn for fn in os.listdir(in_path) if fn.endswith('.jpg')]
n = len(fns)


def f(s):
    img = imread(in_path + s, mode='RGB')
    img = imresize(img, (224, 224))
    return img


pool = ThreadPool(cpu_count())


for i in range(0, n, batch_size):
    k = min(n, i + batch_size)

    batch = pool.map(f, fns[i: k])

    x = np.array(batch, dtype=np.float)
    x = preprocess_input(x)

    y = model.predict(x)

    for j in range(i, k):
        fn = fns[j]
        np.save(out_path + fn + '.npy', y[j-i])

    print(i)
