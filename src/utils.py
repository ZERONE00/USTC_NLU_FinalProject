'''
Author: wxin
Date: 2020-08-09 09:33:54
LastEditTime: 2020-08-09 09:48:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lab/src/utils.py
'''
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os

def read_data(dir='../data/csv/'):
    dfs = []
    for fname in ['train.csv', 'test.csv', 'valid.csv']:
        path = os.path.join(dir, fname)
        df = pd.read_csv(path)
        df = df.sample(frac=1)
        dfs.append(df)
    return dfs

def token_filter(token):
    filter = re.compile(r'[\u4e00-\u9fa5]')
    return len(filter.findall(token)) > 0   

def get_tokens(dir='../data/csv/'):
    dfs = read_data(dir=dir)
    tokens = set()
    for df in dfs:
        for cut_words in tqdm(df['cut_words']):
            tmp_tokens = cut_words.split(' ')
            tokens |= set([token.strip() for token in tmp_tokens if token_filter(token)])
    return tokens

def get_token_id(tokens):
    token2id, id2token = {}, {}
    for token in set(tokens):
        token2id[token] = len(token2id)+1
    for token in token2id:
        id2token[token2id[token]] = token
    
    return token2id, id2token

def get_input(df, token2id):
    x, y = [], df['label']
    for cut_words in tqdm(df['cut_words']):
        tmp_tokens = cut_words.split(' ')
        x.append([token2id[token.strip()] for token in tmp_tokens if token_filter(token)])
    
    return x, y

def padding(x, max_len=500):
    padded_x = []
    for seq in x:
        pseq = seq[:max_len] if len(seq) > max_len else (seq + [0] * (max_len - len(seq)))
        padded_x.append(pseq)
    
    return padded_x