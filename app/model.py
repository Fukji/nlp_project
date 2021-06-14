from fastai.text import *
from utils import *
import numpy as np
import pandas as pd
import os

def load_model():
    path = './app'
    learner = load_learner(path)
    return learner


def load_dict():
    alay_dict = pd.read_csv('./app/colloquial-indonesian-lexicon.csv', usecols=['slang', 'formal'])
    alay_dict = dict(alay_dict.values)
    belanja_dict = pd.read_csv('./app/kamus_belanja.csv', usecols=['slang', 'formal'])
    belanja_dict = dict(belanja_dict.values)
    return alay_dict, belanja_dict


def preprocess(text):
    text = text.lower()
    text = text.replace('[^\w\s]',' ')
    text = text.replace('\s+', ' ')
    text = text.replace(r'(.)\1+',r'\1\1')

    tmp = text.split()
    for i in range(len(tmp)):
        flag = 0
        if tmp[i][-3:] == 'nya':
            tmp[i] = tmp[i][:-3]
            flag = 1
        
        if tmp[i] in alay_dict:
            tmp[i] = alay_dict[tmp[i]]
            
        if tmp[i] in belanja_dict:
            tmp[i] = belanja_dict[tmp[i]]
            
        if flag:
            tmp[i] += ' nya'

    tmp = ' '.join(tmp)
    text = tmp

    text = text.replace('[^\w\s]',' ')
    text = text.replace('\s+', ' ')
    text = text.replace(r'(.)\1+',r'\1\1')

    if len(text.split()) <= 25:
        return text

    return None



def get_prediction(text):
    text = preprocess(text)
    if text is None:
        return "0" , [0, 0, 0, 0, 0]

    prediction = learner.predict(preprocess(text))
    return prediction[1].numpy().tolist(), prediction[2].numpy().tolist()


learner = load_model()
alay_dict, belanja_dict = load_dict()
