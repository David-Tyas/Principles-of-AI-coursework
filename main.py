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
with open("features.json", "w") as file:
    json.dump(features.tolist(), file)
with open("median_incomes.json", "w") as file:
    json.dump(median_incomes.tolist(), file)

plt.style.use("_mpl-gallery")
plt.scatter(median_incomes, median_house_values)