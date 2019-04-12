#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 01:13:14 2019

@author: razat
"""

"""
Importing the libraries
"""
import pickle
import numpy as np
import cv2
import glob
import sys

"""
Importing trained models
"""
filename = 'finalized_model.sav'
classifier = pickle.load(open(filename, 'rb'))
filename = 'finalized_scaler.sav'
scaler = pickle.load(open(filename, 'rb'))

"""
Creating id to label dictionary
"""
labels_training = []
for fruit_dir_path in glob.glob("fruits-360/Training/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    labels_training.append(fruit_label)
labels_training = np.array(labels_training)
label_to_id_dict = {v: i for i,v in enumerate(np.unique(labels_training))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

"""
Loading and preprocessing the argument image
"""
fruit_test = []
image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = cv2.resize(image, (50, 50))
fruit_test.append(image.flatten())
fruit_test = np.array(fruit_test)
fruit_test = scaler.transform(fruit_test)

"""
Prediction
"""
y_pred = classifier.predict(fruit_test)
print(id_to_label_dict[y_pred[0]])
