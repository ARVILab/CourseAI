from scipy.misc import imread, imresize, imsave
import numpy as np
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

in_dir = '../datasets/retouch/input/'
out_dir = '../datasets/retouch/output/'
x_dir = '../datasets/retouch/input_1024/'
y_dir = '../datasets/retouch/output-input_1024/'
z_dir = '../datasets/retouch/output_1024/'

im_size = (1024, 1024)

in_list = [s for s in os.listdir(in_dir) if s.lower().endswith('.jpg')]


def f(ifn):
    ofn = out_dir + ifn
    print(ifn)

    if not os.path.isfile(ofn):
        print('error ' + ifn)
        return

    in_img = imread(in_dir + ifn)
    in_img = imresize(in_img, im_size)

    out_img = imread(ofn)
    out_img = imresize(out_img, im_size)

    diff = np.array(out_img, dtype=np.int16)
    diff = diff - in_img

    imsave(x_dir + ifn, in_img)
    imsave(z_dir + ifn, out_img)
    np.save(y_dir + ifn, diff)


pool = ThreadPool(cpu_count())
pool.map(f, in_list)
