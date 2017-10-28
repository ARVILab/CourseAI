import gensim.models
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import codecs

model = gensim.models.Word2Vec.load('../../datasets/nlp/emb/wordEmbeddings')

words = model.wv.index2word[:10000]
AllVecs = model.wv.syn0[:10000]


# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=AllVecs.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: AllVecs})

# write labels
k = 0
with codecs.open('log/metadata.tsv', 'w', 'utf-8') as f:
    for sent in words:
        f.write(sent.replace('\n', '') + '\n')
        k += 1

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
