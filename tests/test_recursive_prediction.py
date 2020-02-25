import unittest

import numpy as np

from DL.dynamics_learner_interface.dynamics_learner_interface import DynamicsLearnerInterface
from DL.dynamics_learner_interface.dynamics_learner_interface import DynamicsLearnerExample
from tests.fake_data_test_case import FakeDataTestCase


class TestRecursivePrediction(object):

    def test_recursive_prediction(self):
        data = np.load(self.fake_data_npzfile)

        observation_sequences = np.concatenate((data['measured_angles'],
                                                data['measured_velocities'],
                                                data['measured_torques']), 2)

        action_sequences = data['constrained_torques']

        history_length = 10
        prediction_horizon = 3
        dynamics_learner = DynamicsLearnerExample(history_length, 1)
        dynamics_learner.learn(observation_sequences, action_sequences)

        observation_prediction = dynamics_learner.predict_recursively(
                observation_sequences[:, :history_length],
                action_sequences[:, :history_length],
                action_sequences[:, history_length:history_length +
                prediction_horizon - 1])

        rms = np.linalg.norm(observation_sequences[:,
                history_length + prediction_horizon - 1] -
                observation_prediction)
        print('rms: ', rms)


if __name__ == '__main__':
    unittest.main()
