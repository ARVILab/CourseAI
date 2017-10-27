#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'oles'
import tornado
import numpy as np
import gensim.models
import tornado.ioloop
import tornado.web
import json
import os
from sklearn.manifold import TSNE
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

models = {
        'subtitlesEn': {'3d':[],'w2v':[],'startYear':1950,'words':[]},
        'focusRu':{'3d':[],'w2v':[],'startYear':2007,'words':[]}
        }

for i in xrange(1950, 2016, 10):
    if os.path.isfile('../yearModels/subtitlesEn/w2v/'+str(i)):
        m = gensim.models.Word2Vec().load('../yearModels/subtitlesEn/w2v/'+str(i))
        m.syn0 = m.syn0[:8000]
        m.syn1neg = m.syn1neg[:8000]
        models['subtitlesEn']['w2v'].append(m)

    else :
        models['subtitlesEn']['w2v'].append(models['subtitlesEn']['w2v'][-1])

models['subtitlesEn']['words'] = models['subtitlesEn']['w2v'][0].index2word[:7500]
models['subtitlesEn']['3d'] = json.load(open('../yearModels/subtitlesEn/subtitlesEn.json','r'))['years']

for i in range(2007, 2017):
    if os.path.isfile('../yearModels/focusRu/w2v/focus_'+str(i)):
        m = gensim.models.Word2Vec().load('../yearModels/focusRu/w2v/focus_'+str(i))
        m.syn0 = m.syn0[:8000]
        m.syn1neg = m.syn1neg[:8000]
        models['focusRu']['w2v'].append(m)
    else :
        models['focusRu']['w2v'].append(models['focusRu']['w2v'][-1])

models['focusRu']['words'] = models['focusRu']['w2v'][0].index2word[:7500]
models['focusRu']['3d'] = json.load(open('../yearModels/focusRu/focusRu.json','r'))['years']

root = os.path.dirname(__file__)

cache = {}
if os.path.isfile('../yearModels/focusRu/cache.npy'):
    cache['focusRu'] = np.load('../yearModels/focusRu/cache.npy')
else :
    cache['focusRu'] = {}
if os.path.isfile('../yearModels/subtitlesEn/cache.npy'):
    cache['subtitlesEn'] = np.load('../yearModels/subtitlesEn/cache.npy')
else:
    cache['subtitlesEn'] = {}

class ApiHandler(tornado.web.RequestHandler):
    def get(self):
            try:
                keyword = self.get_argument('keyword')
                modelName = self.get_argument('model')
                res = []
                if models[modelName]:
                    try:
                        if keyword in cache[modelName] :
                            res = cache[modelName][keyword].toList()
                        else:
                            for model in models[modelName]['w2v']:
                                yearDist = []
                                for word in model.index2word[:7500]:
                                    yearDist.append(model.similarity(keyword,word))
                                res.append(yearDist)
                            #cache[modelName][keyword] = res
                            #np.save('../yearModels/'+modelName+'/cache.npy', cache[modelName])
                    except AssertionError:
                        self.write("no word in vocab :" + keyword)
                else :
                     self.write("no such model: " + modelName)
                self.write(json.dumps(res))
            except AssertionError:
                self.write("no word in vocab")

class ModelHandler(tornado.web.RequestHandler):
    def get(self):
            try:
                req = self.get_argument('model')
                if models[req]:
                    resp = {'name': req,
                            'startYear': models[req]['startYear'],
                            'years': models[req]['3d'],
                            'words': models[req]['words']}
                    self.write(json.dumps(resp))
                else :
                    self.write("no such model: " + req)

            except AssertionError:
                self.write("no word in vocab")

def make_app():
    return tornado.web.Application([
        (r"/api", ApiHandler),
        (r"/models", ModelHandler),
        ('/static/(.*)', tornado.web.StaticFileHandler, {'path': os.path.join(root, 'static')})
    ])

if __name__ == "__main__":
    app = make_app()
    print('api started')
    app.listen(8080)
    tornado.ioloop.IOLoop.current().start()


