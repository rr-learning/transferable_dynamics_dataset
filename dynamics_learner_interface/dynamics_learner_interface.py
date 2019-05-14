import numpy as np
import sys
import ipdb
import traceback


class DynamicsLearnerInterface(object):

    def __init__(self, history_len, prediction_horizon):
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon

    def learn(self, observation_sequences, action_sequences):
        """
        Parameters
        ----------
        observations_sequences: np-array of shape nSequences x nStepsPerRollout x nStates
                                past state observations
        action_sequences:       np-array of shape nSequences x nStepsPerRollout x nInputs
                                actions taken at the corresponding time points.
        """
        raise NotImplementedError

    def predict(self, observation_history, action_history, action_future):
        """
        Parameters
        ----------
        observation_history:    np-array of shape nPredictionTasks x
                                self.history_len x nStates
                                all states seen by the system.
        action_history:         np-array of shape nPredictionTasks x
                                self.history_len x nInputs
                                all actions seen by the system.
                                The last action corresponds to the action
                                that was applied at the final time step.
        action_future:          np-array of shape nPredictionTasks x
                                self.prediction_horizon - 1 x nInputs
                                actions to be applied to the system. The first
                                action is the action applied one time step after
                                the last action of the corresponding
                                "action_history".
        Outputs
        ----------
        observation_future:     np-array with nStates dimensions corresponding
                                to the state prediction for the particular
                                prediction horizon. The predicted state will be
                                one time step after the last action of the
                                corresponding action_future.
        """
        raise NotImplementedError

    def load(self, filename):
        """
        Parameters
        ----------
        filename:   string used as filename to load a model.
        """
        raise NotImplementedError

    def save(self, filename):
        """
        Parameters
        ----------
        filename:   string used as filename to save a model.
        """
        raise NotImplementedError

    def _check_learning_inputs(self, observation_sequences, action_sequences):
        assert observation_sequences.shape[:2] == action_sequences.shape[:2]
        assert observation_sequences.shape[2] == 9
        assert action_sequences.shape[2] == 3

    # TODO: rewrite for the new interface.
    def _check_prediction_inputs(self, observation_history, action_history, action_future):
        assert observation_history.shape[0] == action_history.shape[0]
        assert observation_history.shape[1] == 9
        assert action_history.shape[1] == 3
        assert action_future.shape[1] == 3

    # TODO: rewrite for the new interface.
    def _check_prediction_outputs(self, action_future, observation_future):
        assert action_future.shape[0] + 1 == observation_future.shape[0]
        assert observation_future.shape[1] == 9


class DynamicsLearnerExample(DynamicsLearnerInterface):

    def learn(self, observation_sequences, action_sequences):
        self._check_learning_inputs(observation_sequences, action_sequences)

    def predict(self, observation_history, action_history, action_future):

        self._check_prediction_inputs(observation_history, action_history, action_future)

        observation_future = np.zeros((action_future.shape[0] + 1,
                observation_history.shape[1]))

        self._check_prediction_outputs(action_future, observation_future)
        return observation_future


if __name__ == '__main__':
    try:

        data = np.load('../../Dataset/dataset_v01.npz')

        observation_sequences = np.concatenate((data['measured_angles'],
                                                data['measured_velocities'],
                                                data['measured_torques']), 2)

        action_sequences = data['constrained_torques']

        dynamics_learner = DynamicsLearnerExample()
        dynamics_learner.learn(observation_sequences, action_sequences)

        observation_future = dynamics_learner.predict(observation_sequences[0, :10],
                                                      action_sequences[0, :10],
                                                      action_sequences[0, 10:15])

        squared_error = np.linalg.norm(observation_sequences[0, 14] - observation_future[4])
        print('squared error: ', squared_error)

        ipdb.set_trace()

    except:
        traceback.print_exc(sys.stdout)
        _, _, tb = sys.exc_info()
        ipdb.post_mortem(tb)
