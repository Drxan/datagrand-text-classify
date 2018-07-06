# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 20:00
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : pretrain_model.py
# @Software: PyCharm
import pandas as pd
from text_utils import data_helper
from text_utils import word2vec
import os
import gc


current_dir = os.path.dirname(__file__)


def train_model(train_data, model_save_path):
    texts = [txt.split() for txt in train_data]
    print('[1-1] Training word2vector model...')
    model = word2vec.train_word2vec(texts)
    print('[1-2] Saving model...')
    model.save(os.path.join(model_save_path, 'char2v.model'))
    return model


if __name__ == '__main__':
    level = 'article'
    train_df = data_helper.load_data(os.path.join(current_dir, 'datas/train_set.csv'), use_cols=[level])
    train_model(train_df[level], os.path.join(current_dir, 'datas/'))
    gc.collect()
