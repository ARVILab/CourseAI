from keras.models import *
from keras.layers import *
from keras.utils.data_utils import get_file
# from keras.optimizers import SGD
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return resized


def VGGUnet(n_classes, input_height=256, input_width=256, vgg_level=4):
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

    embFeatures = Conv2D(512, (3, 3), padding='same')(f5)
    embFeatures = BatchNormalization()(embFeatures)
    embFeatures = Activation('relu')(embFeatures)

    d = UpSampling2D((2, 2))(embFeatures)
    d = Concatenate(axis=-1)([levels[vgg_level-1], d])
    d = Conv2D(512, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)
    seg_4 = Conv2D(n_classes, (1, 1), strides=(1, 1),  name='wv_4_conv', activation='softmax')(d)
    seg_4 = Lambda(Interp, arguments={'shape': (256, 256)}, name='seg_4')(seg_4)

    d = UpSampling2D((2, 2))(d)
    d = Concatenate(axis=-1)([levels[vgg_level-2], d])
    d = Conv2D(512, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)
    seg_3 = Conv2D(n_classes, (1, 1), strides=(1, 1),  name='wv_3_conv', activation='softmax')(d)
    seg_3 = Lambda(Interp, arguments={'shape': (256, 256)}, name='ww_3')(seg_3)

    d = UpSampling2D((2, 2))(d)
    d = Concatenate(axis=-1)([levels[vgg_level-3], d])
    d = Conv2D(256, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)
    seg_2 = Conv2D(n_classes, (1, 1), strides=(1, 1), name='wv_2_conv', activation='softmax')(d)
    seg_2 = Lambda(Interp, arguments={'shape': (256, 256)}, name='ww_2')(seg_2)

    d = UpSampling2D((2, 2))(d)
    d = Concatenate(axis=-1)([levels[vgg_level - 4], d])
    d = Conv2D(128, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)
    seg_1 = Conv2D(n_classes, (1, 1), strides=(1, 1), name='wv_1_conv', activation='softmax')(d)
    seg_1 = Lambda(Interp, arguments={'shape': (256, 256)}, name='ww_1')(seg_1)

    d = UpSampling2D((2, 2))(d)
    d = Conv2D(128, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    seg_0 = Conv2D(n_classes, (1, 1), strides=(1, 1), name='wv_0_conv', activation='softmax')(d)

    model = Model(img_input, [seg_0, seg_1, seg_2, seg_3, seg_4])

    # sgd = SGD(lr=0.001, momentum=0.9)
    model.compile('adam',
                  loss=['categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy'])

    return model


if __name__ == '__main__':
    m = VGGUnet(81)
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='vggUnet.png')
