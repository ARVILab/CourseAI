# -*- coding: utf-8 -*-

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


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
    m.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    m.summary()
    return m


# кількість класів, підставте ваше значення
nb_classes = 42

model = get_model(nb_classes=nb_classes)

# при необхідності завантажити ваги:
# model.load_weights('weights_finetuned.h5')

img_height = 224
img_width = 224
batch_size = 8

# розділити датасет на тренувальний та тестовий
# у пропорції 90/10
train_dir = ''
test_dir = ''

# зробити генератор за рекомендаціями статті:
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    generator=train_generator,
    # validation_data= напишіть генератор для тестових даних
    steps_per_epoch=42,
    nb_epoch=42,
    callbacks=[ModelCheckpoint('weights_finetuned.h5', save_best_only=True, monitor='val_loss')])

model.save_weights('weights_finetuned.h5')
