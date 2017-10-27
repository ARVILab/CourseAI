import numpy as np
import gensim.models
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('training model for year: 2007')
print('loading sentences...')
sentences = gensim.models.word2vec.LineSentence("data/all.txt")

model = gensim.models.Word2Vec(min_count=15,size=200)
model.build_vocab(sentences)

sentences = gensim.models.word2vec.LineSentence("data/years/2007.txt")
model.train(sentences)
model.save('models/focus_2007')

for year in range(2008,2017):
	print('training model for year: ' + str(year))
	print('loading sentences...')
	sentences = gensim.models.word2vec.LineSentence("data/years/"+str(year)+".txt")
	print('training...')
	model.train(sentences)
	model.save('models/focus_'+str(year))
