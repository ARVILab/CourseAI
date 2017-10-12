from __future__ import print_function
from math import ceil
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose
# from keras.optimizers import SGD
import numpy as np
import cv2


learning_rate = 1e-3


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return resized


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv"+lvl+"_" + sub_lvl + "_1x1_reduce",
             "conv"+lvl+"_" + sub_lvl + "_1x1_reduce_bn",
             "conv"+lvl+"_" + sub_lvl + "_3x3",
             "conv"+lvl+"_" + sub_lvl + "_3x3_bn",
             "conv"+lvl+"_" + sub_lvl + "_1x1_increase",
             "conv"+lvl+"_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0], use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad, name=names[2], use_bias=False)(prev)

    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4], use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl+"_" + sub_lvl + "_1x1_proj",
             "conv" + lvl+"_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0], use_bias=False)(prev)
    else:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad, lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added


def ResNet(inp, layers):
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0], use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2], use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4], use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res_128 = relu1

    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i+2)
    res_64 = res
    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i+2)
    res_32 = res
    # 4_1 - 4_6
    res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1, modify_stride=True)
    for i in range(5):
        res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i+2)
    res_16 = res
    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1, modify_stride=True)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i+2)

    res = Activation('relu')(res)
    res_8 = res
    return res_128, res_64, res_32, res_16, res_8


def interp_block(prev_layer, level, feature_map_shape, str_lvl=1):
    str_lvl = str(str_lvl)

    names = [
        "conv5_3_pool"+str_lvl+"_conv",
        "conv5_3_pool"+str_lvl+"_conv_bn"
        ]

    kernel = (10*level, 10*level)
    strides = (10*level, 10*level)
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    prev_layer = Lambda(Interp, arguments={'shape': feature_map_shape})(prev_layer)
    return prev_layer


def build_psp(res, input_shape):
    feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    interp_block1 = interp_block(res, 6, feature_map_size, str_lvl=1)
    interp_block2 = interp_block(res, 3, feature_map_size, str_lvl=2)
    interp_block3 = interp_block(res, 2, feature_map_size, str_lvl=3)
    interp_block6 = interp_block(res, 1, feature_map_size, str_lvl=6)

    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res

def FaceModel( input_shape=(256, 256)):
    inp = Input(input_shape + (3,))
    res_128, res_64, res_32, res_16, res_8 = ResNet(inp, layers=50)
    f_8 = Activation('relu')(BN()(Conv2D(128, kernel_size=(1, 1), padding='same')(res_8)))
    f_16 = Activation('relu')(BN()(Conv2D(128, kernel_size=(1, 1), padding='same')(res_16)))
    f_32 = Activation('relu')(BN()(Conv2D(128, kernel_size=(1, 1), padding='same')(res_32)))
    f_64 = Activation('relu')(BN()(Conv2D(128, kernel_size=(1, 1), padding='same')(res_64)))
    f_128 = Activation('relu')(BN()(Conv2D(128, kernel_size=(1, 1), padding='same')(res_128)))

    d_16 = Activation('relu')(BN()(Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same')(f_8)))
    d_32 = Activation('relu')(BN()(Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same')(Concatenate(axis=-1)([f_16,d_16]))))
    d_64 = Activation('relu')(BN()(Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same')(Concatenate(axis=-1)([f_32, d_32]))))
    d_128 = Activation('relu')(BN()(Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same')(Concatenate(axis=-1)([f_64, d_64]))))
    d_256 = Activation('relu')(BN()(Conv2DTranspose(64, (4, 4), strides=(2,2), padding='same')(Concatenate(axis=-1)([d_128, f_128]))))


    mask = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='conv_mask')(d_256)

    model = Model(inputs=inp, outputs=mask)
    model.load_weights('../weights/pretrained.h5', by_name=True)
    for layer in model.layers[:-56]:
        layer.trainable = False
    # sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


DATA_MEAN = np.array([[[103.939, 116.779, 123.68]]])


def preprocess_image(img, input_width=473):
    float_img = cv2.resize(img, (input_width, input_width), interpolation=cv2.INTER_LINEAR).astype('float16')
    centered_image = float_img - DATA_MEAN
    return centered_image


if __name__ == '__main__':
    m = FaceModel()
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='FaceModel.png')
