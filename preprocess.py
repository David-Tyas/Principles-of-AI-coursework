# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:49:21 2025

@author: David Tyas
"""

import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json

(features, median_house_values) = datasets.fetch_california_housing(return_X_y=True)
median_incomes = features[: , 0]

def format_income(income):
    return f"${(income * 10_000):.2f}"
def format_house_value(house_value):
    return f"${(house_value * 100_000):.2f}"

stats = {
    "Median income": {
        "Max": format_income(np.max(median_incomes)),
        "Min": format_income(np.min(median_incomes)),
        "Mean": format_income(np.mean(median_incomes)),
        "Median": format_income(np.median(median_incomes)),
        "Standard deviation": format_income(np.std(median_incomes))
    },
    "Median house value": {
        "Max": format_house_value(np.max(median_house_values)),
        "Min": format_house_value(np.min(median_house_values)),
        "Mean": format_house_value(np.mean(median_house_values)),
        "Median": format_house_value(np.median(median_house_values)),
        "Standard deviation": format_house_value(np.std(median_house_values))
    }
}
'''with open("statistics.json", "w") as file:
    json.dump(stats, file)
print(f"""Median income statistics:
Max: {format_income(np.max(median_incomes))}
Min: {format_income(np.min(median_incomes))}
Mean: {format_income(np.mean(median_incomes))}
Median: {format_income(np.median(median_incomes))}
Standard deviation: {format_income(np.std(median_incomes))}""", end="\n\n")
print(f"""Median house value statistics:
Max: {format_house_value(np.max(median_house_values))}
Min: {format_house_value(np.min(median_house_values))}
Mean: {format_house_value(np.mean(median_house_values))}
Median: {format_house_value(np.median(median_house_values))}
Standard deviation: {format_house_value(np.std(median_house_values))}""")'''

'''(training_median_incomes,
testing_median_incomes,
training_median_house_values,
testing_median_house_values) = train_test_split(median_incomes, median_house_values, train_size=0.2)

def put_array_in_json_file(arr, file_name):
    with open(file_name, "w") as file:
        json.dump(arr.tolist(), file)
put_array_in_json_file(features, "features.json")
put_array_in_json_file(median_incomes, "median_incomes.json")
put_array_in_json_file(training_median_incomes, "training_median_incomes.json")
training_median_incomes.dump("training_median_incomes.pickle")
put_array_in_json_file(testing_median_incomes, "testing_median_incomes.json")
testing_median_incomes.dump("testing_median_incomes.pickle")
put_array_in_json_file(training_median_house_values, "training_median_house_values.json")
training_median_house_values.dump("training_median_house_values.pickle")
put_array_in_json_file(testing_median_house_values, "testing_median_house_values.json")
testing_median_house_values.dump("testing_median_house_values.pickle")'''

plt.style.use("_mpl-gallery")
'''plt.axis([np.min(median_incomes), np.max(median_incomes),
          np.min(median_house_values), np.max(median_house_values)])'''
plt.xlabel("Median income of district ($10,000s)", wrap=True)
plt.ylabel("Median house value of district ($100,000s)", wrap=True)
plt.scatter(median_incomes, median_house_values, c="#00ff00", edgecolors="#0000ff")