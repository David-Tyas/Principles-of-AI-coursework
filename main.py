# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:49:21 2025

@author: David Tyas
"""

import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import json

(xs, ys) = datasets.fetch_california_housing(return_X_y=True)
with open("xs.json", "w") as file:
    json.dump(xs.tolist(), file)

'''plt.style.use("_mpl-gallery")
plt.scatter(xs,ys)'''