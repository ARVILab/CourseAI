__author__ = 'oles'
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import gensim.models
import logging
import json
import os
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

top_words = 7500


def save3d(vectors3d, name):
    print('saving 3d vectors for model :' + str(name))

    f = open('UI/static/3d/' + str(name) + '.obj', 'w')
    f.write('g language\n')

    for i in range(top_words):
        f.write('v ' +
                str(float(vectors3d[i][0])) + ' ' +
                str(float(vectors3d[i][1])) + ' ' +
                str(float(vectors3d[i][2])) + '\n')
    f.close()


tsnemodel = TSNE(n_components=3, verbose=3, perplexity=42, random_state=0, init="pca", learning_rate=300)

model = gensim.models.Word2Vec().load('UI/models/1950')
with open('UI/static/3d/words.js', 'w') as outfile:
    out = 'words =' + json.dumps(model.index2word[:top_words])
    outfile.write(out)

for year in range(1950, 2005):
    if os.path.isfile('models/' + str(year)):
        model = np.load('models/' + str(year) + '.wv.syn0.npy')[:top_words]
        X = StandardScaler().fit_transform(model)
        # if year > 1905:
        #     tsnemodel.n_iter = 75
        transformedvectors = tsnemodel.fit_transform(X)
        save3d(transformedvectors, str(year))
