from vggFace import VGGFace
from keras.models import Model, Sequential
from keras.layers import TimeDistributed, Conv2D, Flatten, LSTM, Dense, Input, Concatenate, BatchNormalization


def getModel(frameCount=42, nb_classes=7):

    vggFaceEncoder = VGGFace(include_top=False, input_shape=(224, 224, 3))
    for layer in vggFaceEncoder.layers:
        layer.trainable = False

    frameEncoder = Sequential()
    frameEncoder.add(vggFaceEncoder)
    frameEncoder.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='valid'))
    frameEncoder.add(BatchNormalization())
    frameEncoder.add(Flatten())
    frameEncoder.add(Dense(512, activation='relu'))
    frameEncoder.add(BatchNormalization())

    frameInput = Input((frameCount, 224, 224, 3))
    frameFeatures = TimeDistributed(frameEncoder)(frameInput)

    landmarkInput = Input((frameCount, 68, 2))

    landmarkEncoder = Sequential()
    landmarkEncoder.add(Flatten(input_shape=(68, 2)))
    landmarkEncoder.add(Dense(256, activation='relu'))
    landmarkEncoder.add(BatchNormalization())
    landmarkEncoder.add(Dense(128, activation='relu'))
    landmarkEncoder.add(BatchNormalization())

    landmarkFeatures = TimeDistributed(landmarkEncoder)(landmarkInput)

    allFeatures = Concatenate(axis=-1)([frameFeatures, landmarkFeatures])
    lstmFeatures = LSTM(256)(allFeatures)

    out = Dense(nb_classes, activation='sigmoid')(lstmFeatures)

    model = Model(inputs=[frameInput, landmarkInput], outputs=out)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model
