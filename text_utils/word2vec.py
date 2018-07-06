# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 19:56
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : word2vec.py
# @Software: PyCharm
from gensim.models import word2vec


def train_word2vec(texts):
    w2v = word2vec.Word2Vec(sentences=texts, size=100, alpha=0.025, window=5, min_count=1, max_vocab_size=None,
                            sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=1, hs=0, negative=6, cbow_mean=1,
                            iter=60, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False)
    return w2v
