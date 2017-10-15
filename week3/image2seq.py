from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout,Flatten, Input

def imageCapModel(vocab_size=42):
    inp = Input((224,224,3))
    fExtractor = ResNet50(include_top=False, weights='imagenet')
    for layer in fExtractor.layers:
        layer.trainable = False

    features2048 = Flatten()(fExtractor(inp))
    features512 = Dense(512, activation='relu')(features2048)
    repeat12 = RepeatVector(n=12)(features512)
    lstm1 = LSTM(128, return_sequences=True)(repeat12)
    lstm2 = LSTM(128, return_sequences=True)(lstm1)

    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(lstm2)

    model = Model(inputs=inp, outputs=output)
    model.compile('adam', 'categorical_crossentropy')

    return model
