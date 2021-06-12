from flask import Flask, jsonify, request
from fastai.text import *
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from app.model import *


app = Flask(__name__)


@app.route('/')
@app.route('/home')
@app.route('/index')
def home():
    return ('Hello, you\'re not supposed to be here.')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        req = request.form
        label, probs = get_prediction(req['text'])
        return jsonify({
            'label': label,
            'probs': probs
        })


if __name__ == '__main__':
    app.run()
