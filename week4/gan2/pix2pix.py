import keras
from keras import backend as K
from keras import objectives
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from keras.layers.convolutional import Conv2DTranspose
from keras.layers.merge import Concatenate


def concatenate_layers(inputs, concat_axis):
    return Concatenate(axis=concat_axis)(inputs)


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Convolution2D(f,
                         kernel_size=(k, k),
                         padding=border_mode,
                         strides=(s, s),
                         **kwargs)


def Deconvolution(f, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    return Conv2DTranspose(f,
                           kernel_size=(k, k),
                           strides=(s, s),
                           data_format=K.image_data_format(),
                           **kwargs)


def BatchNorm(axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    return BatchNormalization(axis=axis, **kwargs)


def g_unet(in_ch, out_ch, nf, batch_size=1, is_binary=False, name='unet'):
    # type: (int, int, int, int, bool, str) -> keras.models.Model
    """Define a U-Net.
    Input has shape in_ch x 512 x 512
    Parameters:
    - in_ch: the number of input channels;
    - out_ch: the number of output channels;
    - nf: the number of filters of the first layer;
    - is_binary: if is_binary is true, the last layer is followed by a sigmoid
    activation function, otherwise, a tanh is used.
    'channels_first'
    ##>>> unet = g_unet(1, 2, 3, batch_size=5, is_binary=True)
    TheanoShapedU-NET
    ##>>> for ilay in unet.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    ##>>> unet.summary()  #doctest: +NORMALIZE_WHITESPACE
    """
    merge_params = {
        'concat_axis': 3
    }

    i = Input(shape=(256, 256, in_ch))
    print('TensorflowShapedU-NET')

    def get_deconv_shape(samples, channels, x_dim, y_dim):
        return samples, x_dim, y_dim, channels

    merge_params['concat_axis'] = 3

    # in_ch x 256 x 256
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 16 x 16

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 8 x 8

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 4 x 4

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 2 x 2

    conv8 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 1 x 1

    dconv1 = Deconvolution(nf * 8, k=2, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    dconv1 = Dropout(0.5)(dconv1)
    x = concatenate_layers([dconv1, conv7], **merge_params)

    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(nf * 8)(x)
    dconv2 = BatchNorm()(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = concatenate_layers([dconv2, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(nf * 8)(x)
    dconv3 = BatchNorm()(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    x = concatenate_layers([dconv3, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(nf * 8)(x)
    dconv4 = BatchNorm()(dconv4)
    x = concatenate_layers([dconv4, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(nf * 8)(x)
    dconv5 = BatchNorm()(dconv5)
    x = concatenate_layers([dconv5, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 32 x 32

    dconv6 = Deconvolution(nf * 4)(x)
    dconv6 = BatchNorm()(dconv6)
    x = concatenate_layers([dconv6, conv2], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(4 + 4) x 64 x 64

    dconv7 = Deconvolution(nf * 2)(x)
    dconv7 = BatchNorm()(dconv7)
    x = concatenate_layers([dconv7, conv1], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(2 + 2) x 128 x 128

    dconv8 = Deconvolution(nf)(x)
    dconv8 = BatchNorm()(dconv8)
    x = LeakyReLU(0.2)(dconv8)
    # nf*(1 + 1) x 256 x 256

    dconv9 = Deconvolution(out_ch, k=1, s=1)(x)
    # out_ch x 256 x 256

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(dconv9)

    unet = Model(i, out, name=name)

    return unet


def discriminator(a_ch, b_ch, nf, opt=Adam(lr=2e-4, beta_1=0.5), name='d'):
    """Define the discriminator network.
    Parameters:
    - a_ch: the number of channels of the first image;
    - b_ch: the number of channels of the second image;
    - nf: the number of filters of the first layer.
    #>>> K.set_image_dim_ordering('th')
    #>>> disc=discriminator(3,4,2)
    #>>> for ilay in disc.layers: ilay.name='_'.join(ilay.name.split('_')[:-1]) # remove layer id
    #>>> disc.summary() #doctest: +NORMALIZE_WHITESPACE
    """
    i = Input(shape=(256, 256, a_ch + b_ch))

    # (a_ch + b_ch) x 256 x 256
    conv1 = Convolution(nf)(i)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf * 2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(nf * 8)(x)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 16 x 16

    conv5 = Convolution(1)(x)
    out = Activation('sigmoid')(conv5)
    # 1 x 8 x 8

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=opt, loss=d_loss)
    return d


def pix2pix(atob, d, a_ch, b_ch, alpha=100, is_a_binary=False,
            is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5), name='pix2pix'):
    # type: (...) -> keras.models.Model
    """
    Define the pix2pix network.
    :param atob:
    :param d:
    :param a_ch:
    :param b_ch:
    :param alpha:
    :param is_a_binary:
    :param is_b_binary:
    :param opt:
    :param name:
    :return:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 3, 512, 512)       0
    _________________________________________________________________
     (Model)                     (None, 4, 512, 512)       23454
    _________________________________________________________________
    concatenate (Concatenate)    (None, 7, 512, 512)       0
    _________________________________________________________________
     (Model)                     (None, 1, 16, 16)         1813
    =================================================================
    Total params: 25,267.0
    Trainable params: 24,859.0
    Non-trainable params: 408.0
    _________________________________________________________________
    """
    a = Input(shape=(256, 256, a_ch))
    b = Input(shape=(256, 256, b_ch))

    # A -> B'
    bp = atob(a)

    # Discriminator receives the pair of images
    d_in = concatenate_layers([a, bp], concat_axis=-1)

    pix2pix = Model([a, b], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = objectives.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        return L_adv + alpha * L_atob

    # This network is used to train the generator. Freeze the discriminator part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)
    return pix2pix


if __name__ == '__main__':
    unet = g_unet(3, 3, 32, batch_size=8, is_binary=False)
    disc = discriminator(3, 3, 32)
    pp_net = pix2pix(unet, disc, 3, 3)
    print('ololo')
