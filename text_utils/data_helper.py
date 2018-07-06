# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 10:38
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : data_helper.py
# @Software: PyCharm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import gc


def load_data(data_path, use_cols=None, chunk_size=10000):
    df_iter = pd.read_csv(data_path, iterator=True, usecols=use_cols)
    loop = True
    chunks = []
    print('Loading data...')
    while loop:
        try:
            chunk = df_iter.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Loading Finished.")
    df = pd.concat(chunks, ignore_index=True)
    gc.collect()
    return df


def get_indexer(texts, num_words=None):
    tker = Tokenizer(num_words=num_words)
    tker.fit_on_texts(texts)
    return tker
