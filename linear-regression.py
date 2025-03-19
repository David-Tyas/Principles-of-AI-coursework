# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:25:11 2025

@author: David Tyas
"""

import sklearn.linear_model as linear
import matplotlib.pyplot as plt
import numpy as np

training_median_incomes = np.load("training_median_incomes.pickle", allow_pickle=True)
testing_median_incomes = np.load("testing_median_incomes.pickle", allow_pickle=True)
training_median_house_values = np.load("training_median_house_values.pickle", allow_pickle=True)
testing_median_house_values = np.load("testing_median_house_values.pickle", allow_pickle=True)

linear_regression = linear.LinearRegression()
linear_regression.fit(training_median_incomes.reshape(-1,1),training_median_house_values.reshape(-1,1))
prediction = linear_regression.predict(testing_median_house_values.reshape(-1,1))
#print(linear_regression.coef_)

plt.style.use("_mpl-gallery")
plt.scatter(testing_median_incomes, testing_median_house_values, c="#00ff00", edgecolors="#000000")

plt.axline((0.0, linear_regression.intercept_[0]), (1.0, linear_regression.coef_[0][0] + linear_regression.intercept_[0]))