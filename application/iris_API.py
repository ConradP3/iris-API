from flask import Flask,request,render_template

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import pickle


app = Flask(__name__)

# saved the model in pickle format
pipe = pickle.load(open('pipe.pkl','rb'))

feature_cols = ['petal_len', 'petal_width']


@app.route("/")
def getModel():
    return render_template("form.html") 


@app.route('/predict', methods=["POST"])
def predict():
    input_data = []
    for col in feature_cols:
        input_data.append(float(request.form[col]))

    input_df = pd.DataFrame(np.array(input_data).reshape(1,-1), columns=feature_cols)

    loaded_model = pipe.steps[1][1]
    processing = pipe.steps[0][1]
    X = processing.transform(input_df)
    pred = loaded_model.predict(X)

    if pred[0] == 0:
        return 'Iris-Setosa'
    elif pred[0] == 1:
        return 'Iris-Versicolour'
    elif pred[0] == 2:
        return 'Iris-Virginica'
    return 'UNKNOWN'

class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, pandas_df):
        return pandas_df[self.key]


if __name__ == '__main__':
    app.run(host='0.0.0.0')

