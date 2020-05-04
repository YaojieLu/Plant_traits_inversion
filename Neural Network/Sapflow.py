# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:23:03 2019

@author: Yaojie Lu
"""

# Import libraries
import xlrd
import numpy as np
seed = 7
np.random.seed(seed)
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Read xlsx file
workbook = xlrd.open_workbook('../Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
dict = {'Am':'Am_RN_N', 'Nd':'Nd_RN_S', 'Pm':'Pm_RN_S', 'Qc':'Qc_RN_S', 'Qg':'Qg_SS_S', 'Syn':'Synthetic'}
species = 'Pm'#'Am_RN_N' 'Nd_RN_S' 'Pm_RN_S' 'Qc_RN_S' 'Qg_SS_S' 'Synthetic'

# Get data from column with specified colname
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')
D = get_data('D')
X = np.stack([T, I, D], axis=1)
Y = get_data(dict.get(species))

# Define base model
def baseline_model():
	# Create model
	model = Sequential()
	model.add(Dense(24, input_dim = X.shape[1], kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(1, kernel_initializer = 'normal'))
	# Compile model
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	return model

# Evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn = baseline_model, epochs = 50, batch_size = 5, verbose = 0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits = 10, random_state = seed)
results = cross_val_score(pipeline, X, Y, cv = kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# Prediction
pipeline.fit(X, Y)
prediction = pipeline.predict(X)
print(prediction)
pd.DataFrame(prediction).to_csv("../Results/Prediction_%s.csv" % dict.get(species))

# Figures
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot(1, 1, 1)
ax.margins(0.05, 0.05)
ax.plot(list(range(len(Y))), Y, 'r')
ax.plot(list(range(len(Y))), prediction, 'b')
ax.set_xlabel("Days")
ax.set_ylabel("Sap velocity")
ax.set_title(species)
plt.gca().legend(('Observation', 'Prediction'))
fig.tight_layout()
plt.savefig("../Figures/NN_Prediction_%s.png" % dict.get(species))
plt.show()
