from keras.models import Model
from keras.layers import Input, Dense, LSTM, Conv1D, MaxPool1D, Embedding, Bidirectional
import numpy as np
import keras.backend as K
import tensorflow as tf


def getModel(inputlength=512, w2vPath='wordEmbeddings.npy', batch_size=32):
    inp = Input((inputlength,))

    weights = np.load(w2vPath)
    emb = Embedding(len(weights), len(weights[0]), weights=[weights], trainable=False)(inp)
    c1 = Conv1D(128, 3, activation='selu', padding='same')(emb)
    c2 = Conv1D(128, 3, activation='selu', padding='same')(c1)
    m1 = MaxPool1D(pool_size=2)(c2)
    c3 = Conv1D(128, 3, activation='selu', padding='same')(m1)
    m2 = MaxPool1D(pool_size=2)(c3)
    lstm = Bidirectional(LSTM(256, return_sequences=False), merge_mode='concat')(m2)
    docvec = Dense(512, activation='tanh')(lstm)

    def SiamenseLoss(y_true, y_pred):

        firstParts = K.l2_normalize(y_pred[::2, :], axis=-1)
        secondParts = K.l2_normalize(y_pred[1::2, :], axis=-1)

        firstmat = K.expand_dims(firstParts, 1)
        secondmat = K.expand_dims(secondParts, 0)

        firstmat = K.tile(firstmat, (1, batch_size, 1))
        secondmat = K.tile(secondmat, (batch_size, 1, 1))
        cosSim = K.sum((firstmat * secondmat), axis=-1)
        distMatrix = 1 - cosSim

        sameDists = K.mean(tf.diag_part(distMatrix))

        mask = 1 - tf.diag(tf.ones([batch_size]))
        diffSimm = K.mean(K.square(mask * cosSim))

        e = sameDists + diffSimm * 2
        return e

    model = Model(inputs=inp, outputs=docvec)
    model.compile('adam', loss=SiamenseLoss)
    return model
