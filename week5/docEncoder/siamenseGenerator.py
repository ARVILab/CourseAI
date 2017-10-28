import numpy as np


def generator(datapath, batch_size=32, inputLen=256):
    texts = np.load(datapath+'texts.npy')
    np.random.shuffle(texts)
    k = 0
    docCoutnt = len(texts)-1
    while True:
        X = np.zeros((batch_size*2, inputLen), dtype=np.int32)
        Y = np.zeros((batch_size*2, 512), dtype=np.bool)
        n = 0
        for i in range(batch_size):
            dIdx = k % docCoutnt
            if dIdx == 0:
                np.random.shuffle(texts)
            while len(texts[dIdx]) < 200:
                k += 1
                dIdx = k % docCoutnt
                if dIdx == 0:
                    np.random.shuffle(texts)
            txt = texts[dIdx]

            m = int(len(txt) / 2)
            txt1 = txt[:m-np.random.randint(2, 19)]
            txt2 = txt[m-np.random.randint(0, 19):]

            for j in range(min(inputLen, len(txt1))):
                X[n, j] = txt1[j]
            n += 1
            for j in range(min(inputLen, len(txt2))):
                X[n, j] = txt2[j]
            n += 1
            k += 1
        yield X, Y
# g = generator('data/ru/', batch_size=32, inputLen = 256)
# for i in range(0,10000):
#     x, y  =g.next()
#     print i
