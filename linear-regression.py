# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:25:11 2025

@author: David Tyas
"""

import sklearn.linear_model as linear
import numpy as np

training_median_incomes = np.load("training_median_incomes.pickle", allow_pickle=True)
testing_median_incomes = np.load("testing_median_incomes.pickle", allow_pickle=True)
training_median_house_values = np.load("training_median_house_values.pickle", allow_pickle=True)
testing_median_house_values = np.load("testing_median_house_values.pickle", allow_pickle=True)