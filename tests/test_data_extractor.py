# -*- coding: utf-8 -*-

"""
Testing the extraction of data splits from a npz raw data file.
"""

import numpy as np
import os
import unittest

from DL.utils.data_extractor import extract


class Bunch(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestDataLoading(unittest.TestCase):

    def get_fake_data(self):
        keys = ['measured_velocities', 'constrained_torques',
                'measured_torques', 'measured_angles', 'desired_torques']
        nrollouts = 200
        nlen = 15000
        ndof = 3
        data = {}
        for k in keys:
            data[k] = np.arange(nrollouts * nlen * ndof).reshape(
                    (nrollouts, nlen, ndof))
        return data

    def setUp(self):
        self.full_data = self.get_fake_data()
        self.arguments = Bunch(discard_prefix=1000, takeoutrollouts_iid=9,
                takeoutrollouts_validation=3)

    def helper_shape_assert(self, data, shape):
        for k in data.keys():
            self.assertEqual(data[k].shape, shape)

    def helper_flatten_data(self, data):
        array = []
        for k in data.keys():
            array.append(data[k].flatten())
        return np.concatenate(array)

    def test_extract(self):
        training, validation, testiid, testtransfer_sets = extract(
                self.arguments, self.full_data)
        self.assertEqual(training.keys(), self.full_data.keys())
        self.assertEqual(validation.keys(), self.full_data.keys())
        self.assertEqual(testiid.keys(), self.full_data.keys())
        self.helper_shape_assert(training, (38, 14000, 3))
        self.helper_shape_assert(testiid, (9, 14000, 3))
        self.helper_shape_assert(validation, (3, 14000, 3))
        for dataset in testtransfer_sets:
            self.assertEqual(dataset.keys(), self.full_data.keys())
            self.helper_shape_assert(dataset, (9, 14000, 3))
        a = self.helper_flatten_data(training)
        b = self.helper_flatten_data(testiid)
        c = self.helper_flatten_data(validation)
        intersections = np.concatenate([np.intersect1d(a, b),
                np.intersect1d(b, c), np.intersect1d(a, c)])
        self.assertEqual(np.size(intersections), 0)
        intersections = []
        for dataset in testtransfer_sets:
            intersections.append(np.intersect1d(a,
                    self.helper_flatten_data(dataset)))
        intersections = np.concatenate(intersections)
        self.assertEqual(np.size(intersections), 0)


if __name__ == '__main__':
    unittest.main()
