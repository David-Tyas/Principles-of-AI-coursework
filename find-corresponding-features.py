# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 12:32:21 2025

@author: David Tyas
"""

import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import json

(features, median_house_values) = datasets.fetch_california_housing(return_X_y=True)

training_median_incomes = np.load("training_median_incomes.pickle", allow_pickle=True)
testing_median_incomes = np.load("testing_median_incomes.pickle", allow_pickle=True)
training_median_house_values = np.load("training_median_house_values.pickle", allow_pickle=True)
testing_median_house_values = np.load("testing_median_house_values.pickle", allow_pickle=True)

used_feature_row_indices = set()

def find_feature_row(income):
    for (i,row) in enumerate(features):
        if row[0] == income and i not in used_feature_row_indices:
            used_feature_row_indices.add(i)
            return row

def find_corresponding_features(median_incomes):
    features = []
    for income in median_incomes:
        features.append(find_feature_row(income))
    return np.array(features)

training_features = find_corresponding_features(training_median_incomes)
testing_features = find_corresponding_features(testing_median_incomes)
training_features.dump("training_features.pickle")
testing_features.dump("testing_features.pickle")
def put_array_in_json_file(arr, file_name):
    with open(file_name, "w") as file:
        json.dump(arr.tolist(), file)
put_array_in_json_file(training_features, "training_features.json")
put_array_in_json_file(testing_features, "testing_features.json")
