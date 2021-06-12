from fastai.text import *
import numpy as np
from utils import *

def load_model():
    path = '/home/fukji/Documents/pytorch_deploy/nlp_project/app'
    learner = load_learner(path)
    return learner


def preprocess(text):
    return text

def get_prediction(text):
    prediction = learner.predict(preprocess(text))
    return prediction[1].numpy().tolist(), prediction[2].numpy().tolist()
    return result


learner = load_model()