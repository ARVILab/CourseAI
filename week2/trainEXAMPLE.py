from __future__ import print_function

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import random
import cv2
import numpy as np
import glob

from skimage.transform import rotate

img_rows = 160
img_cols = 224

smooth = 1.


def augmentation(image, imageB, org_width=160, org_height=224, width=190, height=262):
    max_angle = 20
    image = cv2.resize(image, (width, height))
    imageB = cv2.resize(imageB, (width, height))

    angle = np.random.randint(max_angle)
    if np.random.randint(2):
        angle = -angle
    image = rotate(image, angle, resize=True)
    imageB = rotate(imageB, angle, resize=True)

    xstart = np.random.randint(width - org_width)
    ystart = np.random.randint(height - org_height)
    image = image[ystart:ystart + org_height, xstart:xstart + org_width]
    imageB = imageB[ystart:ystart + org_height, xstart:xstart + org_width]

    if np.random.randint(2):
        image = cv2.flip(image, 1)
        imageB = cv2.flip(imageB, 1)

    # if np.random.randint(2):
    #     image = cv2.flip(image, 0)
    #     imageB = cv2.flip(imageB, 0)

    image = cv2.resize(image, (org_width, org_height))
    imageB = cv2.resize(imageB, (org_width, org_height))

    return image, imageB


def get_unet():
    inputs = Input((img_cols, img_rows, 3))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)

    # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

    up6 = merge(
        [Convolution2D(256, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv5)), conv4],
        mode='concat', concat_axis=-1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge(
        [Convolution2D(128, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv6)), conv3],
        mode='concat', concat_axis=-1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge(
        [Convolution2D(64, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv7)), conv2],
        mode='concat', concat_axis=-1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge(
        [Convolution2D(32, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv8)), conv1],
        mode='concat', concat_axis=-1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model


imgWidth = 473


labelPath = 'data/masks/'
imgsPath = 'data/imgs/'
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
        k = k % imgCount
        rgb1Batch = np.zeros((batch_size, img_cols, img_rows, 3))
        labelWVBatch = np.zeros((batch_size, img_cols, img_rows, 1), dtype=np.float32)

        for b in range(0, batch_size):
            if not k:
                random.shuffle(imgNames)
            rgb1Batch[b, :, :, :], labelWVBatch[b, :, :, 0] = prepareImage(k)
            k += 1
        yield rgb1Batch, labelWVBatch


traingen = generator(imgList[:trainCount], batch_size=8)
testgen = generator(imgList[trainCount:], batch_size=8)


model = get_unet()
model.load_weights('./weights/unet.hdf5')

k = 0
for imgpath in imgList:
    img, img1 = augmentation(cv2.imread(imgpath), cv2.imread(imgpath))
    lbl = model.predict(np.expand_dims(img, axis=0))[0]
    b_channel, g_channel, r_channel = cv2.split((img*255).astype(np.uint8))

    alpha_channel = (lbl[:, :, 0] * 255).astype(np.uint8)  # creating a dummy alpha channel image.
    img_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    cv2.imwrite('res/' + str(k) + '.png', img_RGBA)
    k += 1

model.summary()
model.fit_generator(
    generator=traingen, validation_data=testgen,
    steps_per_epoch=500,
    validation_steps=50,
    epochs=30000,
    verbose=1,
    callbacks=[
        ModelCheckpoint('./weights/unet.hdf5', verbose=1, monitor='val_loss', save_best_only=False)
    ])


