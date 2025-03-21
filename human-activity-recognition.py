# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:58:57 2025

@author: David Tyas
"""

import sklearn.svm as svm
import matplotlib.pyplot as plt
import numpy as np

training_xs = []
activities = []

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

'''with open(r".\UCI HAR dataset\train\subject_train.txt", "r") as file:
    i = 0
    while True:
        line = file.readline()
        if line == "":
            break
        assert line[-1] == "\n"
        line = line[:-1]
        assert i < len(training_xs)
        training_xs[i].append(int(line))
        i += 1'''

with open(r".\UCI HAR dataset\train\y_train.txt", "r") as file:
    while True:
        line = file.readline()
        if line == "":
            break
        assert line[-1] == "\n"
        activities.append(1 if int(line[:-1]) > 3 else 0)
