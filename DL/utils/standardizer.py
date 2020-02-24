# -*- coding: utf-8 -*-

import numpy as np

class Standardizer:

    def __init__(self, data=None):
        """
        initializes the standardizer. The standardizer will standardize
        vectors of dimension "dataDimension" component-wise

        Parameters
        ----------
        data:   nPoints x dataDimension
                training data to obtain empirical variance and mean
        """
        if data is not None:
            self.updateParameters(data)

    def updateParameters(self, data):
        n = 0
        accum = None
        squared = None
        for vector in data:
            if accum is None:
                accum = vector.copy()
                squared = vector * vector
            else:
                accum += vector
                squared += vector * vector
            n += 1
        self.means = accum / n
        self.stds = np.sqrt(squared / n - self.means * self.means)

    def standardize(self, dataVector):
        """
        standardizes a vector of dimension "dataDimension" component wise using
        the empirical mean and std of this Standardizer instance.

        dataVector can be either a vector or an array of shape
        nDataPoints x dataDimension
        """
        if dataVector.ndim == 1:
            return (dataVector - self.means) / self.stds
        elif dataVector.ndim == 2:
            return (dataVector - self.means) / self.stds
        raise Exception("Wrong input format")

    def unstandardize(self, dataVector):
        """
        retransforms a standardized vector of dimension "dataDimension" component
        wise using the empirical mean and std of this Standardizer instance.

        dataVector can be either a vector or an array of shape
        nDataPoints x dataDimension
        """
        if dataVector.ndim == 1:
            return (dataVector*self.stds) + self.means
        elif dataVector.ndim == 2:
            return (dataVector*self.stds) + self.means
        raise Exception("Wrong input format")
