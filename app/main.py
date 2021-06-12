from flask import Flask, jsonify, request
from app.model import *


app = Flask('_name__')


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
