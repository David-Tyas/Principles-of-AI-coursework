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
training_features = np.load("training_features.pickle", allow_pickle=True)
testing_features = np.load("testing_features.pickle", allow_pickle=True)
training_median_house_values = np.load("training_median_house_values.pickle", allow_pickle=True)
testing_median_house_values = np.load("testing_median_house_values.pickle", allow_pickle=True)

def mean_square_error(model, xs, ys):
    total = 0.0
    prediction = model.predict(xs)
    for (i, predicted_y) in enumerate(prediction):
        total += (predicted_y - ys[i])**2
    return total/len(ys)

med_inc_lin_reg = linear.LinearRegression()
# np.ndarray.reshape(-1, 1) is used,
# because LinearRegression.fit() is designed
# for x and y being 2D.
med_inc_lin_reg.fit(training_median_incomes.reshape(-1,1), training_median_house_values.reshape(-1,1))
print(f"Score for just using median income: {med_inc_lin_reg.score(testing_median_incomes.reshape(-1,1), testing_median_house_values.reshape(-1,1))}")
example_median_income = 8.0
example_prediction = med_inc_lin_reg.predict(np.array(example_median_income).reshape(1,-1))[0][0]
print(f"""Predicted median house value for median income of {format_income(example_median_income)}:
{format_house_value(example_prediction)}""")
#print(linear_regression.coef_)

plt.style.use("_mpl-gallery")
plt.scatter(testing_median_incomes, testing_median_house_values, c="#00ff00", edgecolors="#000000")
plt.xlabel("Median income of district ($10,000s)", wrap=True)
plt.ylabel("Median house value of district ($100,000s)", wrap=True)

y_intercept = med_inc_lin_reg.intercept_[0]
second_y_value = med_inc_lin_reg.coef_[0][0] + med_inc_lin_reg.intercept_[0]
assert (y_intercept ==
        med_inc_lin_reg.predict(np.array(0.0).reshape(1,-1))[0][0])
assert (second_y_value ==
        med_inc_lin_reg.predict(np.array(1.0).reshape(1,-1))[0][0])
plt.axline((0.0, med_inc_lin_reg.intercept_[0]),
           (1.0, med_inc_lin_reg.coef_[0][0] + med_inc_lin_reg.intercept_[0]))

improved_lin_reg = linear.LinearRegression()
improved_lin_reg.fit(training_features, training_median_house_values)
print(f"Score for using all features: {improved_lin_reg.score(testing_features, testing_median_house_values.reshape(-1,1))}")
