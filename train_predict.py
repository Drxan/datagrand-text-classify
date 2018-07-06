# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 13:08
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : train_predict.py
# @Software: PyCharm
from text_utils import data_helper
import os
from text_utils import data_helper
from keras.preprocessing.sequence import pad_sequences

current_dir = os.path.dirname(__file__)
train_df = data_helper.load_data(os.path.join(current_dir, 'datas/train_set.csv'),use_cols=['article','classify'])
indexer = data_helper.get_indexer(train_df['article'])
x = indexer.texts_to_sequences(train_df['article'])
x = pad_sequences(x,)
y = train_df['classify'].astype(int)-1
