# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:21:26 2023

@author: pedro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

# PREDICTING HEART FAILURE
# For this problem we propose a neural network approach
# trying to reach good performance and clean code:)

# ------- LOADING DATA --------

data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print(data.head)

# exploring the data we realize it´s better to 
# shuffle the rows and then proceede with train_test_split()

data = data.sample(frac=1)
print(data.shape)

#Min max scaler
columns_scale = data.columns
scaler = MinMaxScaler()
data[columns_scale] = scaler.fit_transform(data[columns_scale])

ax = sns.countplot(x= data["DEATH_EVENT"])
ax.bar_label(ax.containers[0])

labels = data.iloc[:,12:13]
features = data.iloc[:,0:12]

# ------- DATA SPLITTING --------

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=5)

print('Traning data:', x_train.shape, y_train.shape)
print('Testing data:', x_test.shape, y_test.shape)

# Helper Function
def training_plot(metrics, history):
  f, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
  for idx, metric in enumerate(metrics):
    ax[idx].plot(history.history[metric], ls='dashed')
    ax[idx].set_xlabel("Epochs")
    ax[idx].set_ylabel(metric)
    ax[idx].legend([metric, 'val_' + metric])

# -------- CREATING NEURAL NETWORK MODEL ----------

# This time we are going to try a simple MLP model
# from here we can get an idea of the performance and we can add more layers later

def build_train_MLP(x_train, y_train, x_test, y_test, loss, weights):
    
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation='relu', input_dim=12),
        keras.layers.Dense(1, activation='sigmoid')
        ])
    
    model.compile(loss=loss, optimizer='sgd', metrics=['accuracy'])
    history_1 = model.fit(x_train, y_train, epochs=70)
    training_plot(['loss', 'accuracy'], history_1)
    print(model.evaluate(x_test, y_test))
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred) 
    print("Classification Report: \n", classification_report(y_test, y_pred))
    
    return y_pred

history = build_train_MLP(x_train, y_train, x_test, y_test, 'mean_squared_error', -1)

# From start we can observe that the data is imbalanced, so we have to deal
# with this first and then trying to get a decent model for prediction

# ------------- HANDLING DATA IMBALANCE --------------------
#!pip install imblearn
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(x_train, y_train)

# some issues with RandomUnderSampler using spyder but it work´s

x_train, x_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2, random_state=15)

history_2 = build_train_MLP(x_train, y_train, x_test, y_test, 'mean_squared_error', -1)


# ---------------- THE END -----------------