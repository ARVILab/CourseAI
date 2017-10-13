from vggFace import VGGFace
from keras.models import Model, Sequential
from keras.layers import TimeDistributed, Conv2D, Flatten, LSTM, Dense, Input


def getModel(frameCount=42, nb_classes=7):

    vggFaceEncoder = VGGFace(include_top=False, input_shape=(224,224,3))
    for layer in vggFaceEncoder.layers:
        layer.trainable = False

    frameEncoder = Sequential()
    frameEncoder.add(vggFaceEncoder)
    frameEncoder.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='valid'))
    frameEncoder.add(Flatten())
    frameEncoder.add(Dense(512, activation='relu'))

    inp = Input((frameCount, 224, 224, 3))
    seqFeatures = TimeDistributed(frameEncoder)(inp)
    lstmFeatures = LSTM(128)(seqFeatures)
    out = Dense(nb_classes, activation='softmax')(lstmFeatures)

    model = Model(inputs=inp, outputs=out)
    model.compile('adam', 'categorical_crossentropy')
    return model

getModel()
