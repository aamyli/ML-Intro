# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:21:06 2020

@author: Amy
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #initializes flask app
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # to render results onto html page
    int_features = [int (x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text="Employee Salary should be $ {}".format(output))

app.run(debug=True)