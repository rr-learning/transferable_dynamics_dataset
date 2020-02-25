# -*- coding: utf-8 -*-

from DL.utils.standardizer import Standardizer
import numpy as np


testData = np.arange(10).reshape([-1, 2])
testData[:, 1] = 2*testData[:, 1]

stateStandardizer = Standardizer(testData)

print(stateStandardizer.means)
print(stateStandardizer.stds)

# test vector
print("standardizing vector")
testVector = testData[3, :]
print(testVector)
standVector = stateStandardizer.standardize(testVector)
print(standVector)
print(stateStandardizer.unstandardize(standVector))

print("standardizing matrix")
print(testData)
standData = stateStandardizer.standardize(testData)
print(standData)
print(stateStandardizer.unstandardize(standData))
print(np.mean(stateStandardizer.standardize(testData)))
print(np.std(stateStandardizer.standardize(testData)))
