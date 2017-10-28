import gensim.models

print('loading sentences...')
sentences = gensim.models.word2vec.LineSentence("../../datasets/nlp/all.txt")

model = gensim.models.Word2Vec(min_count=15,size=200)
model.build_vocab(sentences)

model.train(sentences)
model.save('wordEmbeddings')
print('done')