# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:11:36 2020

@author: Amy
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

x = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

# splits training and test set (this case train with all data)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting model with training data
regressor.fit(x, y)

# saves model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))
