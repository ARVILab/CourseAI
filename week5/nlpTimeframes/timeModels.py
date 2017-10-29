import numpy as np
import gensim.models
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('loading sentences...')
sentences = gensim.models.word2vec.LineSentence("../../datasets/nlp/subs/concat/all.txt")

model = gensim.models.Word2Vec(min_count=15,size=200)
model.build_vocab(sentences)


for year in range(1950,2014):
	print('training model for year: ' + str(year))
	print('loading sentences...')
	sentences = gensim.models.word2vec.LineSentence('../../datasets/nlp/subs/concat/%s.txt' % year)
	print('training...')

	tempModel  = gensim.models.Word2Vec(min_count=15,size=1)
	tempModel.build_vocab(sentences)

	model.train(sentences, total_examples=tempModel.corpus_count, epochs=tempModel.iter)
	model.save('models/'+str(year))
