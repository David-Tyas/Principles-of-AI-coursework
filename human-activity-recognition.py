# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:58:57 2025

@author: David Tyas
"""

import sklearn.decomposition as decomp
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as preprocessing
import sklearn.svm as svm
import matplotlib.pyplot as plt
import numpy as np
import json

training_xs = []
testing_xs = []
training_ys = []
testing_ys = []

with open(r".\UCI HAR dataset\train\X_train.txt", "r") as file:
    while True:
        line = file.readline()
        if line == "":
            break
        assert line[-1] == "\n"
        #Each line starts with two spaces,
        #so we strip them off as well as the line feed.
        line = line[2:-1]
        number_tokens = line.split(" ")
        current_measurements = []
        for token in number_tokens:
            if token != "":
                current_measurements.append(float(token))
        training_xs.append(current_measurements)
        '''print(number_tokens)
        training_xs.append([float(x) for x in filter(lambda token: token != "", number_tokens)])'''

with open(r".\UCI HAR dataset\test\X_test.txt", "r") as file:
    while True:
        line = file.readline()
        if line == "":
            break
        assert line[-1] == "\n"
        #Each line starts with two spaces,
        #so we strip them off as well as the line feed.
        line = line[2:-1]
        number_tokens = line.split(" ")
        current_measurements = []
        for token in number_tokens:
            if token != "":
                current_measurements.append(float(token))
        testing_xs.append(current_measurements)

#with open(r".\UCI HAR dataset\train\subject_train.txt", "r") as file:
#    i = 0
#    while True:
#        line = file.readline()
#        if line == "":
#            break
#        assert line[-1] == "\n"
#        line = line[:-1]
#        assert i < len(training_xs)
#        training_xs[i].append(int(line))
#        i += 1

with open(r".\UCI HAR dataset\train\y_train.txt", "r") as file:
    while True:
        line = file.readline()
        if line == "":
            break
        assert line[-1] == "\n"
        training_ys.append(1 if int(line[:-1]) > 3 else 0)

with open(r".\UCI HAR dataset\test\y_test.txt", "r") as file:
    while True:
        line = file.readline()
        if line == "":
            break
        assert line[-1] == "\n"
        testing_ys.append(1 if int(line[:-1]) > 3 else 0)

mean = np.mean(training_xs, axis=0)
assert len(mean) == len(training_xs[0])
print(mean)

'''svc = svm.LinearSVC()
svc.fit(training_xs, training_ys)
print(svc.score(testing_xs, testing_ys))'''
linear_svc = svm.SVC(kernel="linear")
linear_svc.fit(training_xs, training_ys)
print(linear_svc.score(testing_xs, testing_ys))

def calculate_confusion_matrix(model, xs, ys):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for (i, x) in enumerate(xs):
        y = ys[i]
        if model.predict(np.array(x).reshape(1, -1)) == y:
            if y == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y == 1:
                fn += 1
            else:
                fp += 1
    assert tp + fp + tn + fn == len(xs)
    return [[tp, fp], [tn, fn]]
print(calculate_confusion_matrix(linear_svc, testing_xs, testing_ys))
