import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import os

#Creating the flask app
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model = joblib.load('https://github.com/arjyo851/crop_determination/blob/main/Crop-Recommender.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prediction = prediction[0]

    return render_template('index.html', prediction_text='The recommended crop is {}'.format(prediction))
    

if __name__ == "__main__":
    app.run(debug=True)