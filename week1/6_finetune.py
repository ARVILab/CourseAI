# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import random
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
# from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread, imresize


# модель = ResNet50 без голови з одним dense шаром для класифікації об'єктів на nb_classes
def get_model(cls=100, input_shape=(224, 224, 3)):
    feature_extractor = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    flat = Flatten()(feature_extractor.output)
    # додамо кілька dense шарів:
    dense = Dense(1024, activation='relu')(flat)
    drop = Dropout(0.2)(dense)
    # останній шар - класифікація на 1 клас
    dense = Dense(cls, activation='softmax')(drop)
    m = Model(inputs=feature_extractor.input, outputs=dense)

    # "заморозимо" всі шари ResNet50, крім кількох останніх
    # базові ознаки згорткових шарів перших рівнів досить універсальні, тому ми не будемо міняти їх ваги
    # кількість шарів, які ми "заморожуємо" - це гіперпараметр
    for layer in m.layers[:-3]:
        layer.trainable = False

    # для finetuning ми спочатку використаємо Adam,
    # а потім звичайний SGD з малою швидкістю навчання та моментом
    m.compile(
        optimizer='adam',  # SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    m.summary()

    return m


# шлях до датасету
dataset_path = '../datasets/classifieds/'

dirs = os.listdir(dataset_path)
dirs = filter(lambda x: re.match(r'\d+_', x), dirs)

# кількість класів
nb_classes = len(dirs)

# завантажуємо шляхи до зображень
dataset = []
for d in dirs:
    imgs = [d + '/' + s for s in os.listdir(dataset_path + d) if s.endswith('.jpg')]
    dataset += imgs

random.shuffle(dataset)

# x - 4-вимірний тензор (кількість зображень у батчі, 224, 224, 3)
# y - 2-вимірний тензор (кількість зображень у батчі, кількість класів)
# y[i] = (0, ..., 0, 1, 0, ..., 0) "one hot encoding"


# ділимо датасет на train та validation
split = int(0.1 * len(dataset))
train_dataset = dataset[split:]
val_dataset = dataset[0: split]

model = get_model(cls=nb_classes)

# при необхідності завантажити ваги:
model.load_weights('weights_finetuned.h5')

image_shape = (224, 224, 3)
batch_size = 64


datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


def gen(files):
    n = 0
    while True:
        x = np.zeros((batch_size,) + image_shape, dtype=np.float)
        y = np.zeros((batch_size, nb_classes), dtype=np.float16)
        for i in range(batch_size):
            while True:
                try:
                    img = imread(dataset_path + files[n], mode='RGB')
                    break
                except IOError:
                    continue

            x[i] = imresize(img, image_shape)

            cls = int(files[n].split('_')[0])
            y[i][cls] = 1

            n = (n + 1) % len(files)

            if not n:
                random.shuffle(files)

        # x = datagen.flow(x, batch_size=batch_size).next()
        # x = x.astype(np.float) / 127.5 - 1
        x = preprocess_input(x)

        yield x, y


model.fit_generator(
    generator=gen(train_dataset),
    validation_data=gen(val_dataset),
    steps_per_epoch=1000,
    validation_steps=100,
    nb_epoch=42,
    use_multiprocessing=True,
    callbacks=[ModelCheckpoint('weights_finetuned.h5', save_best_only=True, monitor='val_loss')])

model.save_weights('weights_finetuned.h5')
