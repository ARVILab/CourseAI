from __future__ import print_function

from model import FaceModel
from keras.callbacks import ModelCheckpoint

import random
import cv2
import numpy as np
import glob

from skimage.transform import rotate
np.random.seed(1)
random.seed(1)

img_rows = 256
img_cols = 256

smooth = 1.

DATA_MEAN = np.array([[[103.939, 116.779, 123.68]]])
def augmentation(image, imageB, org_width=256, org_height=256, width=300, height=300):
    max_angle = 20
    image = cv2.resize(image, (width, height))
    imageB = cv2.resize(imageB, (width, height), interpolation=cv2.INTER_NEAREST)

    angle = np.random.randint(max_angle)
    if np.random.randint(2):
        angle = -angle
    image = rotate(image, angle, resize=True) * 255
    imageB = rotate(imageB, angle, resize=True)

    xstart = np.random.randint(width - org_width)
    ystart = np.random.randint(height - org_height)
    image = image[ystart:ystart + org_height, xstart:xstart + org_width]
    imageB = imageB[ystart:ystart + org_height, xstart:xstart + org_width]

    if np.random.randint(2):
        image = cv2.flip(image, 1)
        imageB = cv2.flip(imageB, 1)

    image = cv2.resize(image, (org_width, org_height), interpolation=cv2.INTER_LINEAR).astype('float16') - DATA_MEAN
    imageB = cv2.resize(imageB, (org_width, org_height), interpolation=cv2.INTER_NEAREST)

    return image, imageB


imgWidth = 256

testMode = False

labelPath = 'data/masks/'
#labelPath = '/DATA/FACES/data/masks/'
imgsPath = 'data/imgs/'
#imgsPath = '/DATA/FACES/data/imgs/'
imgList = [f for f in glob.glob(imgsPath + '*')]

random.shuffle(imgList)
trainCount = int(len(imgList) * 0.95)


def generator(imgNames, batch_size=8):
    imgCount = len(imgNames)

    def prepareImage(imgId):
        filename = imgNames[imgId]
        lblPath = filename.replace('.jpg', '.npy').replace(imgsPath, labelPath)

        lbl = np.load(lblPath).astype(np.float)
        img = cv2.imread(filename)
        if len(img.shape) < 3:
            canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float)
            canvas[:, :, 0] = img
            canvas[:, :, 1] = img
            canvas[:, :, 2] = img
            img = canvas
        img, lbl = augmentation(img, lbl)
        return img, lbl
    k = 0
    while 1:
        rgb1Batch = np.zeros((batch_size, img_cols, img_rows, 3))
        labelWVBatch = np.zeros((batch_size, img_cols, img_rows, 1), dtype=np.float32)
        for b in range(0, batch_size):
            k = k % imgCount
            if not k:
                random.shuffle(imgNames)
            rgb1Batch[b, :, :, :], labelWVBatch[b, :, :, 0] = prepareImage(k)
            k += 1
        yield rgb1Batch, labelWVBatch

batch_size = 4

traingen = generator(imgList[:trainCount], batch_size=batch_size)
testgen = generator(imgList[trainCount:], batch_size=batch_size)

testCount = len(imgList[trainCount:])

model = FaceModel()
model.summary()
model.load_weights('./weights/faceModel.hdf5')
if not testMode:
    model.fit_generator(
        generator=traingen, validation_data=testgen,
        steps_per_epoch=int(trainCount/batch_size),
        validation_steps=int(testCount/batch_size),
        epochs=30000,
        verbose=1,
        callbacks=[
            ModelCheckpoint('./weights/faceModel.hdf5', verbose=1, monitor='val_loss', save_best_only=False)
        ])
else:
    k = 0
    for imgpath in imgList[trainCount:]:
        img = cv2.resize(cv2.imread(imgpath),(imgWidth,imgWidth)) - DATA_MEAN
        x, y = testgen.next()
        lbl = model.predict(np.expand_dims(x[0], axis=0))[0]
        b_channel, g_channel, r_channel = cv2.split((x[0] + DATA_MEAN).astype(np.uint8))

        alpha_channel = (lbl[:, :, 0] * 255).astype(np.uint8)
        img_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        cv2.imwrite('res/' + str(k) + '.png', img_RGBA)
        k += 1


