import numpy as np

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Concatenate
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import cv2


batch_size = 128


# discriminator
def discriminator_model():
    _inp = Input(shape=(32, 32, 3))
    x = Conv2D(64, 3, strides=1, padding='same')(_inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x_f = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x_f)

    x_label = Dense(41, activation='softmax')(x_f)

    return Model(inputs=_inp, outputs=[x, x_label])


def generator_model():
    _inp = Input(shape=(100,))
    _inp_label = Input((41,))

    x_label = Dense(8 * 8 * 128)(_inp_label)
    x_label = Activation('relu')(x_label)
    x_label = Reshape((8, 8, 128))(x_label)
    x_label = Dropout(0.4)(x_label)

    x = Dense(8 * 8 * 256)(_inp)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Reshape((8, 8, 256))(x)
    x = Dropout(0.4)(x)

    x = Concatenate(axis=-1)([x, x_label])

    x = UpSampling2D()(x)
    x = Conv2DTranspose(256, 5, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, 3, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(128, 3, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, 3, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, 3, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, 3, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(3, 3, padding='same')(x)
    x = Activation('tanh')(x)

    return Model(inputs=[_inp, _inp_label], outputs=x)


def custom_D_loss_mse(y_true, y_pred):
    shape = tf.shape(y_pred)
    half_size = tf.cast(shape[0]/2, tf.int32)
    realImages_true = y_true[:half_size, :]
    realImages_pred = y_pred[:half_size, :]

    mse = K.mean(K.square(realImages_true - realImages_pred))
    return mse


discriminator = discriminator_model()
discriminator.compile(
    loss=['binary_crossentropy', custom_D_loss_mse],
    optimizer=RMSprop(lr=0.0002, decay=6e-8))

generator = generator_model()

inp_z = Input((100,))
inp_l = Input((41,))
g = generator([inp_z, inp_l])
d = discriminator(g)
adversarial = Model(inputs=[inp_z, inp_l], outputs=d)
adversarial.compile(
    loss=['binary_crossentropy', 'mse'],
    optimizer=RMSprop(lr=0.0001, decay=3e-8))


inconNames = np.load('../../datasets/icons/X.npy')
X = []
for icon_name in inconNames:
    img = cv2.imread('../../datasets/icons/smallIcons/' + icon_name + '.png')
    img = img.astype(np.float)/127.5 - 1
    X.append(img[:, :, :3])

x_train = np.asarray(X)
x_labels = np.load('../../datasets/icons/Y.npy')
idxs = np.random.randint(0, x_train.shape[0], size=16)
test_lbls = x_labels[idxs, :]


def train(train_steps=200000, batch_size=256, save_interval=0):
    noise_input = None

    if save_interval > 0:
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])

    for i in range(train_steps):
        idxs = np.random.randint(0, x_train.shape[0], size=batch_size)
        images_train = x_train[idxs, :, :, :]
        lbls_train = x_labels[idxs, :]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

        images_fake = generator.predict([noise, lbls_train])
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2 * batch_size, 1])
        y_lbls = np.repeat(lbls_train, 2, axis=0)

        y[batch_size:, :] = 0
        d_loss = discriminator.train_on_batch(x, [y, y_lbls])

        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        a_loss = adversarial.train_on_batch([noise, lbls_train], [y, lbls_train])

        log_mesg = "%d: [D loss: %f, classLoss: %f]" % (i, d_loss[1], d_loss[2])
        log_mesg = "%s  [A loss: %f, classLoss: %f]" % (log_mesg, a_loss[1], a_loss[2])
        print(log_mesg)

        if save_interval > 0:
            if (i + 1) % save_interval == 0:
                plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))
                adversarial.save_weights('g_d_icons.h5')


def plot_images(save2file=False, fake=True, samples=16, noise=None, step=0):
    filename = 'icons.png'
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "data/icons_%d.png" % step
        images = generator.predict([noise, test_lbls])
    else:
        i = np.random.randint(0, x_train.shape[0], samples)
        images = x_train[i, :, :, :]

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = (images[i, :, :, ::-1]+1.) * 127.5
        image = np.reshape(image, [32, 32, 3]).astype(np.uint8)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    train(train_steps=200000, batch_size=batch_size, save_interval=100)
    plot_images(fake=True)
    plot_images(fake=False, save2file=True)
