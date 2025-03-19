# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:25:11 2025

@author: David Tyas
"""

import sklearn.linear_model as linear
import matplotlib.pyplot as plt
import numpy as np

def format_income(income):
    return f"${(income * 10_000):.2f}"
def format_house_value(house_value):
    return f"${(house_value * 100_000):.2f}"

training_median_incomes = np.load("training_median_incomes.pickle", allow_pickle=True)
testing_median_incomes = np.load("testing_median_incomes.pickle", allow_pickle=True)
training_median_house_values = np.load("training_median_house_values.pickle", allow_pickle=True)
testing_median_house_values = np.load("testing_median_house_values.pickle", allow_pickle=True)

linear_regression = linear.LinearRegression()
linear_regression.fit(training_median_incomes.reshape(-1,1),training_median_house_values.reshape(-1,1))
prediction = linear_regression.predict(testing_median_house_values.reshape(-1,1))
example_median_income = 8.0
example_prediction = linear_regression.predict(np.array(example_median_income).reshape(-1,1))[0][0]
print(f"""Predicted median house value for median income of {format_income(example_median_income)}:
{format_house_value(example_prediction)}""")
#print(linear_regression.coef_)

plt.style.use("_mpl-gallery")
plt.scatter(testing_median_incomes, testing_median_house_values, c="#00ff00", edgecolors="#000000")

y_intercept = linear_regression.intercept_[0]
second_y_value = linear_regression.coef_[0][0] + linear_regression.intercept_[0]
assert y_intercept == linear_regression.predict(np.array(0.0).reshape(-1,1))[0][0]
assert second_y_value == linear_regression.predict(np.array(1.0).reshape(-1,1))[0][0]
plt.axline((0.0, linear_regression.intercept_[0]), (1.0, linear_regression.coef_[0][0] + linear_regression.intercept_[0]))