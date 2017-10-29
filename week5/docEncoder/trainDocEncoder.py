from siamenseGenerator import generator
from docEncoder import getModel
from keras.callbacks import ModelCheckpoint

batch_size = 128
datapath = 'data/'


model = getModel(inputlength=256, w2vPath='wordEmbeddings.npy', batch_size=batch_size)

trainGen = generator(datapath=datapath, batch_size=batch_size, inputLen=256)


model.fit_generator(trainGen, steps_per_epoch = 2500, epochs=1424342, callbacks=[
    ModelCheckpoint('weights/DocEncoder.h5', monitor='loss', save_best_only=False)
])