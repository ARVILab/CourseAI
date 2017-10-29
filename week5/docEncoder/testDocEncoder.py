import numpy as np
from siamenseGenerator import generator
from docEncoder import getModel
from sklearn.metrics.pairwise import cosine_distances
import pylab as plt

batch_size = 32
datapath = 'data/'


model = getModel(inputlength=256, w2vPath='data/wordEmbeddings.h5.npy')

trainGen = generator(datapath=datapath, batch_size=32, inputLen=256)

model.load_weights('weights/DocEncoder.h5')

for i in range(100):
    x, y = trainGen.next()
    res = model.predict(x)

    f = res[::2]
    s = res[1::2]
    mat = cosine_distances(f, s)

    plt.imshow(mat, interpolation="none", cmap='Blues')
    tick_marks = np.arange(len(x[::2]))
    plt.xticks(tick_marks, [np.count_nonzero(d) for d in x[::2]], rotation=45)
    plt.yticks(tick_marks, [np.count_nonzero(d) for d in x[1::2]])
    plt.show()

print('ololo')
