from keras.layers.core import Layer, Activation,  Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras import models


def segnet(nClasses, optimizer=None, input_height=256, input_width=256):
    kernel = 3
    filter_size = 64
    pool_size = 2

    model = models.Sequential()
    model.add(Layer(input_shape=(input_height, input_width, 3)))

    # encoder
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(128, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # decoder
    model.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(nClasses, 1, 1, border_mode='same'))

    model.outputHeight = model.output_shape[-3]
    model.outputWidth = model.output_shape[-2]

    model.add(Reshape((nClasses, model.outputHeight * model.outputWidth),
                      input_shape=(nClasses, model.outputHeight, model.outputWidth)))

    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))

    if optimizer is not None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    m = segnet(81, optimizer='adam')
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='modelGraphs/segnet.png')

