# -*- coding: utf-8 -*-

import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# кількість класів, підставте ваше значення
nb_classes = 42


# модель = ResNet50 без голови з одним dense шаром для класифікації об'єктів на nb_classes
def get_model(nb_classes=100):
    feature_extractor = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    flat = Flatten()(feature_extractor.output)
    # можна додати кілька dense шарів:
    # d = Dense(nb_classes*2, activation='relu')(flat)
    # d = Dense(nb_classes, activation='softmax')(d)
    d = Dense(nb_classes, activation='softmax')(flat)
    m = Model(inputs=feature_extractor.input, outputs=d)

    # "заморозимо" всі шари ResNet50, крім кількох останніх
    # базові ознаки згорткових шарів перших рівнів досить універсальні, тому ми не будемо міняти їх ваги
    # кількість шарів, які ми "заморожуємо" - це гіперпараметр
    for layer in m.layers[:-12]:
        layer.trainable = False

    # для finetuning ми використаємо звичайний SGD з малою швидкістю навчання та моментом
    m.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    m.summary()
    return m


model = get_model(nb_classes=nb_classes)
import random
X = np.random.random((512,224,224,3))
y = np.zeros((512,800))

for i in range(0,512):
    y[i,random.randint(0,799)] = 1

model.fit(X,y, batch_size=16, validation_split=0.05, nb_epoch=42,
          callbacks=[ModelCheckpoint('weights_finetuned.h5', save_best_only=True, monitor='val_loss')])
model.save_weights('weights_finetuned.h5')
print('ololo')