from videoModel import getModel
import numpy as np
from keras.callbacks import ModelCheckpoint




X = np.load('/DATA/CourseAI/datasets/facs/X.npy')
Y = np.load('/DATA/CourseAI/datasets/facs/Y.npy')

minLength = 10000
for x in X:
    if len(x[0]) < minLength:
        minLength = len(x[0])
print minLength
AllSeqIds = np.arange(len(X))
np.random.shuffle(AllSeqIds)

seqLength = 6

trainCount = int(len(AllSeqIds)*0.9)

def generator(seqIds, batch_size=2):
    imgCount = len(seqIds)
    k = 0
    while 1:
        framesBatch = np.random.random((batch_size, seqLength, 224, 224, 3))
        landmarksBatch = np.random.random((batch_size, seqLength, 68, 2))

        labelsBatch = np.zeros((batch_size, 65), dtype=np.float32)
        for b in range(0, batch_size):
            # while Y[seqIds[k]][1] is None:
            #     k+=1
            #     k = k % imgCount
            #     if not k:
            #         np.random.shuffle(seqIds)
            #     k = min(imgCount-1,k)
            k += 1
            k = k % imgCount
            if not k:
                np.random.shuffle(seqIds)

            seqId = seqIds[k]
            allFrames = X[seqId][0]
            if len(allFrames)-seqLength > 0:
                startFrame = np.random.randint(0, len(allFrames)-seqLength)
                startFrame = len(allFrames)-seqLength
            else:
                startFrame = 0

            seqImg = X[seqId][0][startFrame:startFrame + seqLength]
            seqLand = X[seqId][1][startFrame:startFrame + seqLength]

            if len(seqImg) < seqLength:
                for j in range(0,seqLength):
                    framesBatch[b, j, :, :, :] = seqImg[j%len(seqImg),:,:,:]
                    landmarksBatch[b, j, :, :] = seqLand[j%len(seqImg),:,:]
            else:
                framesBatch[b,:,:,:,:] = seqImg
                landmarksBatch[b, :, :, :] = seqLand[:, :, :]

            labelsBatch[b, :] = Y[seqId][0]
            k += 1
            k = min(imgCount - 1, k)
        framesBatch[:, :, :, :, 0] -= 93.5940
        framesBatch[:, :, :, :, 1] -= 104.7624
        framesBatch[:, :, :, :, 2] -= 129.1863
        yield [framesBatch, landmarksBatch], labelsBatch

batch_size = 16

traingen = generator(AllSeqIds[:trainCount], batch_size=batch_size)
testgen = generator(AllSeqIds[trainCount:], batch_size=batch_size)

testCount = len(AllSeqIds[trainCount:])

model = getModel(frameCount=seqLength, nb_classes=65)
model.summary()
#model.load_weights('./weights/facsModel.hdf5')
model.fit_generator(
    generator=traingen, validation_data=testgen,
    steps_per_epoch=int(trainCount/batch_size),
    validation_steps=int(testCount/batch_size),
    epochs=5,
    verbose=1,
    callbacks=[
        ModelCheckpoint('./weights/facsModel.hdf5', verbose=1, monitor='val_loss', save_best_only=False)
    ])
print('ololo')
