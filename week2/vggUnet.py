from keras.models import *
from keras.layers import *
from keras.utils.data_utils import get_file

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGGUnet(n_classes, input_height=256, input_width=256, vgg_level=3):
    img_input = Input(shape=(input_height, input_width, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x
    vgg = Model(img_input, x)
    weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    vgg.load_weights(weights_path, by_name=True)
    for layer in vgg.layers:
        layer.trainable = False

    levels = [f1, f2, f3, f4, f5]

    d = levels[vgg_level]

    d = Conv2D(512, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    d = UpSampling2D((2, 2))(d)
    d = Concatenate(axis=-1)([levels[vgg_level-1], d])
    d = Conv2D(256, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    d = UpSampling2D((2, 2))(d)
    d = Concatenate(axis=-1)([levels[vgg_level - 2], d])
    d = Conv2D(128, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    d = UpSampling2D((2, 2))(d)
    d = Concatenate(axis=-1)([levels[vgg_level - 3], d])
    d = Conv2D(64, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    d = Conv2D(n_classes, (1, 1))(d)
    d = Activation('softmax')(d)

    finalmodel = Model(inputs=img_input, outputs=d)

    return finalmodel


if __name__ == '__main__':
    m = VGGUnet(81)
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='modelGraphs/vggUnet.png')