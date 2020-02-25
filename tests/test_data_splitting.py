# -*- coding: utf-8 -*-

"""
Testing basic functionalities of DL.utils.data_splitting.py
"""

import unittest

import numpy as np

from DL.utils.data_loading import loadRobotData
from DL.utils.data_splitting import CompleteRolloutsDataSplitter
from tests.fake_data_test_case import FakeDataTestCase


class TestDataSplitting(FakeDataTestCase):

    def test_default_load(self):
        data_splitter = CompleteRolloutsDataSplitter(self.fake_data_npzfile,
                np.arange(5))
        training_obs, training_act = data_splitter.get_training_data()
        testing_obs, testing_act = data_splitter.get_test_data()
        self.assertEqual(training_obs.shape[0], training_act.shape[0])
        self.assertEqual(training_obs.shape[1], testing_obs.shape[1])
        self.assertEqual(training_act.shape[1], testing_act.shape[1])
        self.assertEqual(training_obs.shape[0], training_act.shape[0])
        self.assertEqual(testing_obs.shape[0], testing_act.shape[0])
        flag = np.array_equal(np.sort(np.concatenate((
                data_splitter.train_rollouts,
                data_splitter.test_rollouts))),
                np.arange(self.observations.shape[0]))
        self.assertTrue(flag)

