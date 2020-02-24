"""
Training/validation data splitters.
"""
import numpy as np
from DL.utils.data_loading import loadRobotData


class DataSplitterInterface(object):

    def get_training_data(self):
        """
        Returns two sets of sequences as training data.

        Outputs
        -------
        training_observation_sequences:  np-array of shape:
                                         nSequences x nStepsPerRollout x nStates

        training_action_sequences:       np-array of shape:
                                         nSequences x nStepsPerRollout x nInputs
        """
        raise NotImplementedError

    def get_test_data(self):
        """
        Returns two sets of sequences as testing data.

        Outputs
        -------
        testing_observation_sequences:  np-array of shape:
                                        nSequences x nStepsPerRollout x nStates

        testing_action_sequences:       np-array of shape:
                                        nSequences x nStepsPerRollout x nInputs
        """
        raise NotImplementedError


class CompleteRolloutsDataSplitter(DataSplitterInterface):

    def __init__(self, data_filename, test_rollout_indexes):
        self.observations, self.actions = loadRobotData(data_filename)
        self.test_rollouts = np.unique(test_rollout_indexes)
        nrollouts = self.observations.shape[0]
        self.train_rollouts = np.setdiff1d(np.arange(nrollouts),
                self.test_rollouts)
        assert self.test_rollouts.shape == test_rollout_indexes.shape,\
                "There are repeated numbers in the provided array."
        assert self.train_rollouts.size + self.test_rollouts.size == nrollouts

    def get_training_data(self):
        return self.observations[self.train_rollouts],\
                self.actions[self.train_rollouts]

    def get_test_data(self):
        return self.observations[self.test_rollouts],\
                self.actions[self.test_rollouts]

