import gensim.models
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('loading sentences...')
sentences = gensim.models.word2vec.LineSentence("../../datasets/nlp/subs/en/all.txt")

model = gensim.models.Word2Vec(min_count=15, size=200)
model.build_vocab(sentences)


for year in range(1950, 2005):
    print('training model for year: ' + str(year))
    print('loading sentences...')
    with open('../../datasets/nlp/subs/timeframeFile.txt', 'w') as timeframeFile:
        for j in range(year-2, year + 3):
            fn = '../../datasets/nlp/subs/en/%s.txt' % j
            if os.path.isfile(fn):
                with open(fn, 'r') as f:
                    timeframeFile.write(f.read())

    sentences = gensim.models.word2vec.LineSentence('../../datasets/nlp/subs/timeframeFile.txt')
    print('training...')

    tempModel = gensim.models.Word2Vec(min_count=15, size=1)
    tempModel.build_vocab(sentences)

    model.train(sentences, total_examples=tempModel.corpus_count, epochs=tempModel.iter)
    model.save('UI/models/' + str(year))
