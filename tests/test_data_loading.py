# -*- coding: utf-8 -*-

"""
Testing basic functionalities of DL.utils.dataLoading.py
"""

import unittest

import numpy as np

from DL.utils.data_loading import loadRobotData, unrollTrainingData
from DL.utils.data_loading import unrollTrainingDataStream, computeNumberOfTrainingPairs
from tests.fake_data_test_case import FakeDataTestCase


class TestDataLoading(FakeDataTestCase):

    def test_unrolling0(self):
        hist_len = 1
        pred_horizon = 1
        targets, inputs = unrollTrainingData(self.observations, self.actions,
                hist_len, pred_horizon, False)
        ntargets, dimtargets = targets.shape
        ninputs, diminputs = inputs.shape
        self.assertEqual(ntargets, ninputs)
        self.assertEqual(dimtargets, 9)
        self.assertEqual(diminputs, 12)

    def test_unrolling1(self):
        hist_len = 2
        pred_horizon = 2
        targets, inputs = unrollTrainingData(self.observations, self.actions,
                hist_len, pred_horizon, False)
        ntargets, dimtargets = targets.shape
        ninputs, diminputs = inputs.shape
        self.assertEqual(ntargets, ninputs)
        self.assertEqual(dimtargets, 9)
        self.assertEqual(diminputs, 27)

    def print_unrolling(self, difference_learning):
        print("Unrolling test with difference learning {}".format(
            difference_learning))
        testObs = np.arange(2*3*4).reshape([2,3,4])
        testActions = np.arange(2*3*4).reshape([2,3,4])+100
        hist_len, pred_horizon = (1, 1)
        old_testObs, old_testActions = testObs.copy(), testActions.copy()
        targets, inputs = unrollTrainingData(testObs, testActions, hist_len,
                pred_horizon, difference_learning)
        self.assertTrue(np.array_equal((old_testObs, old_testActions),
                (testObs, testActions)))
        print("test observations")
        print(testObs)
        print("test actions")
        print(testActions)
        print("first observation sequence")
        print(testObs[0, :, :])
        print("first action sequence")
        print(testActions[0, :, :])
        print("inputs")
        print(inputs)
        print("targets")
        print(targets)

    def test_unrolling2(self):
        self.print_unrolling(False)
        self.print_unrolling(True)

    def test_unroll_minibatching(self):
        hist_len = 1
        pred_horizon = 1
        for average in (False, True):
            targets, inputs = unrollTrainingData(self.observations,
                    self.actions, hist_len, pred_horizon, False,
                    average=average)
            minibatch_generator = unrollTrainingDataStream(self.observations,
                    self.actions, hist_len, pred_horizon, False,
                    average=average, shuffle=False, infinite=False)
            joint_inputs = []
            joint_targets = []
            for minibatch_targets, minibatch_inputs in minibatch_generator:
                joint_inputs.append(minibatch_inputs)
                joint_targets.append(minibatch_targets)
            self.assertTrue(np.array_equal(targets, np.vstack(joint_targets)))
            self.assertTrue(np.array_equal(inputs, np.vstack(joint_inputs)))

            minibatch_generator = unrollTrainingDataStream(self.observations,
                    self.actions, hist_len, pred_horizon, False,
                    average=average, shuffle=True, infinite=False)
            joint_inputs = []
            joint_targets = []
            for minibatch_targets, minibatch_inputs in minibatch_generator:
                joint_inputs.append(minibatch_inputs)
                joint_targets.append(minibatch_targets)
            joint_targets = np.vstack(joint_targets)
            joint_inputs = np.vstack(joint_inputs)
            self.assertEqual(joint_targets.shape, targets.shape)
            self.assertEqual(joint_inputs.shape, inputs.shape)
            self.assertFalse(np.array_equal(targets, joint_targets))
            self.assertFalse(np.array_equal(inputs, joint_inputs))
            self.assertTrue(np.array_equal(
                np.sort(joint_targets.flatten()), np.sort(targets.flatten())))
            self.assertTrue(np.array_equal(
                np.sort(joint_inputs.flatten()), np.sort(inputs.flatten())))

    def test_dataset_size(self):
        history_len, prediction_horizon = (1, 10)
        dataset_size = computeNumberOfTrainingPairs(self.observations,
                                                    history_len,
                                                    prediction_horizon)
        stream_size = sum(1 for _ in unrollTrainingDataStream(self.observations,
                                                              self.actions,
                                                              history_len,
                                                              prediction_horizon,
                                                              True,
                                                              infinite=False))
        self.assertEqual(dataset_size, stream_size)

    def test_infinite_data_stream(self):
        hist_len = 1
        pred_horizon = 1
        finite_data_stream = unrollTrainingDataStream(self.observations,
                self.actions, hist_len, pred_horizon, False, shuffle=False,
                infinite=False)
        finite_data_stream_2 = unrollTrainingDataStream(self.observations,
                self.actions, hist_len, pred_horizon, False, shuffle=False,
                infinite=False)
        infinite_data_stream = unrollTrainingDataStream(self.observations,
                self.actions, hist_len, pred_horizon, False, shuffle=False,
                infinite=True)
        for y, x in finite_data_stream:
            yy , xx = next(infinite_data_stream)
            self.assertTrue(np.array_equal(y, yy))
            self.assertTrue(np.array_equal(x, xx))

        # The infinite stream starts again from the beginning.
        for y, x in finite_data_stream_2:
            yy , xx = next(infinite_data_stream)
            self.assertTrue(np.array_equal(y, yy))
            self.assertTrue(np.array_equal(x, xx))


if __name__ == '__main__':
    unittest.main()
