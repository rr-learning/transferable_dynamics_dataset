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
        self.means = np.mean(data, axis=0)
        self.stds = np.std(data, axis=0)

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
