# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:58:57 2025

@author: David Tyas
"""

import sklearn.svm as svm
import matplotlib.pyplot as plt
import numpy as np

activities = []

with open(r".\UCI HAR dataset\train\y_train.txt", "r") as file:
    while True:
        line = file.readline()
        if line == "":
            break
        assert line[-1] == "\n"
        activities.append(1 if int(line[:-1]) > 3 else 0)

print(activities)