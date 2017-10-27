from docEncoder import getModel
import numpy as np
import pickle
model = getModel(inputlength=256, w2vPath='data/wordEmbeddings.npy')
model.load_weights('weights/DocEncoder.h5')

datapath = 'data/'
texts = np.load(datapath+'texts.npy')
vocabs = pickle.load(open(datapath + 'indexedVocab.pkl', 'rb'))

testTexts = np.random.choice(texts, size=5000)

testTxt = []
batch_size = 128
AllVecs = []
for i in range(0, len(testTexts), batch_size):
    batch_size = min(batch_size, len(testTexts) - i)
    X = np.zeros((batch_size, 256), dtype=np.int32)
    for j in range(0, batch_size):
        txtIdxs = testTexts[i+j]
        plainTxt = []
        for k in range(0,min(256, len(txtIdxs))):
            X[j,k] = txtIdxs[k]
            plainTxt.append(vocabs['index2word'][txtIdxs[k]])
        plainTxt = ' '.join(plainTxt)
        testTxt.append(plainTxt)
    vecs = model.predict(X, batch_size=batch_size)
    for v in vecs:
        AllVecs.append(v)

AllVecs = np.array(AllVecs)


import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import codecs


# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=AllVecs.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: AllVecs})

# write labels
k=0
with codecs.open('log/log/metadata.tsv', 'w', 'utf-8') as f:
    for sent in testTxt:
        f.write(sent.replace('\n','') + '\n')
        k+=1

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter('log', sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

# save the model
saver = tf.train.Saver()
saver.save(sess, os.path.join('log', "model.ckpt"))



