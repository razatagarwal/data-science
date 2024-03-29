#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Importing Libraries
"""
import numpy as np
import glob
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import time
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

"""
Creating object for Standard Scaler
"""
scaler = StandardScaler()

"""
Loading and preprocessing the training dataset
"""
fruit_training = []
labels_training = []
for fruit_dir_path in glob.glob("fruits-360/Training/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (50, 50))
        fruit_training.append(image.flatten())
        labels_training.append(fruit_label)        
fruit_training = np.array(fruit_training)
labels_training = np.array(labels_training)
fruit_training = scaler.fit_transform(fruit_training)
label_to_id_dict = {v: i for i,v in enumerate(np.unique(labels_training))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids_training = np.array([label_to_id_dict[x] for x in labels_training])

"""
Loading and preprocessing the test dataset
"""
fruit_test = []
labels_test = []
for fruit_dir_path in glob.glob("fruits-360/Test/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (50, 50))
        fruit_test.append(image.flatten())
        labels_test.append(fruit_label)
fruit_test = np.array(fruit_test)
fruit_test = scaler.transform(fruit_test)
labels_test = np.array(labels_test)
label_to_id_dict = {v: i for i,v in enumerate(np.unique(labels_test))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids_test = np.array([label_to_id_dict[x] for x in labels_test])

"""
PCA 
"""
pca = PCA(0.95)
pca.fit(fruit_training)
fruit_training = pca.transform(fruit_training)
fruit_test = pca.transform(fruit_test)

"""
K Nearest Neighbours Classifier
"""
start_time = time.time()
classifier = KNeighborsClassifier(n_neighbors = 1,
                                  metric = 'minkowski',
                                  p = 2,
                                  algorithm = 'auto',
                                  weights = 'distance')
classifier = classifier.fit(fruit_training, label_ids_training)
print("--- %s seconds ---" % (time.time() - start_time))

"""
Prediction
"""
y_pred = classifier.predict(fruit_test)
precision = accuracy_score(y_pred, label_ids_test) * 100
print("Accuracy with K Nearest Neighbors: {0:.6f}".format(precision))

"""
Confusion Matrix
"""
cm = confusion_matrix(label_ids_test, y_pred)

"""
Save the model to disk
"""
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
filename = 'finalized_scaler.sav'
pickle.dump(scaler, open(filename, 'wb'))