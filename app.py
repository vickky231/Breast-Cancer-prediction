import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = [
        'mean radius','mean texture','mean perimeter','mean area',
        'mean smoothness','mean compactness','mean concavity',
        'mean concave points','mean symmetry','mean fractal dimension',
        'radius error','texture error','perimeter error','area error',
        'smoothness error','compactness error','concavity error',
        'concave points error','symmetry error','fractal dimension error',
        'worst radius','worst texture','worst perimeter','worst area',
        'worst smoothness','worst compactness','worst concavity',
        'worst concave points','worst symmetry','worst fractal dimension'
    ]

    df = pd.DataFrame(features_value, columns=features_name)

    output = model.predict(df)

    if output[0] == 0:
        res_val = "breast cancer"
    else:
        res_val = "no breast cancer"

    return render_template(
        'index.html',
        prediction_text=f'Patient has {res_val}'
    )


if __name__ == "__main__":
    app.run()

