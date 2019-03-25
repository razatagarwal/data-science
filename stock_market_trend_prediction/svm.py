#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:55:34 2019

@author: razat
"""

"""
Importing Libraries
"""
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

"""
Creating object for Standard Scaler
"""
scaler = StandardScaler()

"""
Loading the dataset
"""
dataset = pd.read_csv("DATA/AMAZON_5495_DATASET.csv")
length = len(dataset)

"""
Train test split
"""
split = int(length * 0.8)
x_train = dataset.iloc[0:split, 0:17].values
x_test = dataset.iloc[split:, 0:17].values
y_train = dataset.iloc[0:split, 17].values
y_test = dataset.iloc[split:, 17].values

"""
Fitting StandardScaler to data
"""
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""
Training the model
"""
start_time = time.time()
classifier = svm.SVC()
classifier = classifier.fit(x_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

"""
Prediction
"""
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

"""
Calculation Accuracy
"""
accuracy = accuracy_score(y_pred, y_test) * 100
print("Accuracy with SVM: {0:.6f}".format(accuracy))

"""
Calculating F-measure
"""
f_measure = f1_score(y_test, y_pred) * 100
print("F-measure with SVM: {0:.6f}".format(f_measure))