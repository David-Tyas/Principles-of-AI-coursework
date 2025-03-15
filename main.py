# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:49:21 2025

@author: David Tyas
"""

import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import json

(features, median_house_values) = datasets.fetch_california_housing(return_X_y=True)
median_incomes = features[: , 0]

def format_income(income):
    return f"${(income * 10_000):.2f}"
def format_house_value(house_value):
    return f"${(house_value * 100_000):.2f}"

print(f"""Median income statistics:
Max: {format_income(np.max(median_incomes))}
Min: {format_income(np.min(median_incomes))}
Mean: {format_income(np.mean(median_incomes))}
Median: {format_income(np.median(median_incomes))}""", end="\n\n")
print(f"""Median house value statistics:
Max: {format_house_value(np.max(median_house_values))}
Min: {format_house_value(np.min(median_house_values))}
Mean: {format_house_value(np.mean(median_house_values))}
Median: {format_house_value(np.median(median_house_values))}""")

'''with open("features.json", "w") as file:
    json.dump(features.tolist(), file)
with open("median_incomes.json", "w") as file:
    json.dump(median_incomes.tolist(), file)'''

plt.style.use("_mpl-gallery")
plt.scatter(median_incomes, median_house_values)