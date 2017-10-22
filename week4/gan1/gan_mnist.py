import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt


# discriminator
def discriminator_model():
    _inp = Input(shape=(28, 28, 1))
    x = Conv2D(64, 5, strides=2, padding='same')(_inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=_inp, outputs=x)


def generator_model():
    _inp = Input(shape=(100,))
    x = Dense(7 * 7 * 256)(_inp)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Reshape((7, 7, 256))(x)
    x = Dropout(0.4)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(128, 5, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(64, 5, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, 5, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(1, 5, padding='same')(x)
    # no batch norm here
    x = Activation('sigmoid')(x)

    return Model(inputs=_inp, outputs=x)


discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0002, decay=6e-8), metrics=['accuracy'])

generator = generator_model()

inp = Input((100,))
g = generator(inp)
d = discriminator(g)
adversarial = Model(inputs=inp, outputs=d)
adversarial.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001, decay=3e-8), metrics=['accuracy'])

x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)


def train(train_steps=2000, batch_size=256, save_interval=0):
    noise_input = None

    if save_interval > 0:
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])

    for i in range(train_steps):
        images_train = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

        images_fake = generator.predict(noise)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2 * batch_size, 1])

        y[batch_size:, :] = 0
        d_loss = discriminator.train_on_batch(x, y)

        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        a_loss = adversarial.train_on_batch(noise, y)

        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)

        if save_interval > 0:
            if (i + 1) % save_interval == 0:
                plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))


def plot_images(save2file=False, fake=True, samples=16, noise=None, step=0):
    filename = 'mnist.png'
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "data/mnist_%d.png" % step
        images = generator.predict(noise)
    else:
        i = np.random.randint(0, x_train.shape[0], samples)
        images = x_train[i, :, :, :]

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    train(train_steps=2000, batch_size=256, save_interval=200)
    plot_images(fake=True)
    plot_images(fake=False, save2file=True)
