import numpy as np
import sys
import ipdb
import traceback

from DL.utils import unrollTrainingData, concatenateActionsStates, Standardizer


class DynamicsLearnerInterface(object):

    def __init__(self, history_length, prediction_horizon):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.observation_dimension = 9
        self.action_dimension = 3

    # do not override this function!
    def learn(self, observation_sequences, action_sequences):
        self._check_learning_inputs(observation_sequences, action_sequences)
        targets, inputs = unrollTrainingData(observation_sequences, action_sequences,
                self.history_length, self.prediction_horizon)

        # Whitening the inputs.
        self.targets_standardizer = Standardizer(targets)
        self.inputs_standardizer = Standardizer(inputs)
        standardized_targets = self.targets_standardizer.standardize(targets)
        standardized_inputs = self.inputs_standardizer.standardize(inputs)

        self._learn(standardized_inputs, standardized_targets)

    # do not override this function!
    def predict(self, observation_history, action_history, action_future=None):
        if action_future is None:
            assert self.prediction_horizon == 1
            action_future = np.empty((observation_history.shape[0],
                                      0,
                                      self.action_dimension))
        self._check_prediction_inputs(observation_history, action_history, action_future)

        # Making a single input from all the input parameters.
        dynamics_inputs = concatenateActionsStates(action_history,
                observation_history, action_future)

        # Whitening the input.
        whitened_input = self.inputs_standardizer.standardize(dynamics_inputs)

        whitened_predictions = self._predict(whitened_input)

        # Dewhitening the output
        dewhitened_predictions = self.targets_standardizer.unstandardize(
                whitened_predictions)

        self._check_prediction_outputs(observation_history,
                dewhitened_predictions)
        return dewhitened_predictions

    # do not override this function!
    def predict_recursively(self, observation_history, action_history, action_future):
        assert self.prediction_horizon == 1
        assert observation_history.shape[1] == self.history_length
        assert action_history.shape[1] == self.history_length

        observation_history_t = observation_history
        action_history_t = action_history
        predicted_observation = self.predict(observation_history_t, action_history_t)

        for t in xrange(action_future.shape[1]):
            predicted_observation = np.expand_dims(predicted_observation, axis=1)
            observation_history_t = np.append(observation_history_t[:, 1:],
                    predicted_observation, axis=1)
            action_history_t = np.append(action_history_t[:, 1:],
                    action_future[:, t:t + 1], axis=1)
            predicted_observation = self.predict(observation_history_t, action_history_t)

            assert (action_history_t[:, :-(t + 1)] == action_history[:, t + 1:]).all()
            assert (observation_history_t[:, :-(t + 1)] == observation_history[:, t + 1:]).all()
            assert (action_history_t[:, -1] == action_future[:, t]).all()
            assert (observation_history_t[:, -1] == predicted_observation).all()

        return predicted_observation

    # override this function
    def name(self):
        raise NotImplementedError

    # override this function
    def _learn(self, training_inputs, training_targets):
        """
        Parameters
        ----------
        training_inputs:        np-array of shape nTrainingInstances x input dim
                                that represents the input to the dynamics
                                (i.e. relevant observations and actions within
                                the history length and prediction horizon)
        training_targets:       np-array of shape nTrainingInstances x state dim
                                denoting the targets of the dynamics model.
        """
        raise NotImplementedError

    # override this function
    def _predict(self, single_input):
        """
        Parameters
        ----------
        single_input:           one dimensional np-array with all the inputs to
                                the dynamics model concatenated (size: input dim)
                                (i.e. relevant observations and actions within
                                the history length and prediction horizon)
        Outputs
        ----------
        observation_prediction: np-array of shape n_samples x observation_dimension
                                corresponding the prediction for the observation
                                prediction_horizon steps after the last observation
                                of observation_history

        """
        raise NotImplementedError

    def _check_learning_inputs(self, observation_sequences, action_sequences):
        assert observation_sequences.shape[:2] == action_sequences.shape[:2]
        assert observation_sequences.shape[2] == self.observation_dimension
        assert action_sequences.shape[2] == self.action_dimension

    def _check_prediction_inputs(self, observation_history, action_history, action_future):
        n_samples = observation_history.shape[0]

        assert observation_history.shape == (n_samples,
                                             self.history_length,
                                             self.observation_dimension)

        assert action_history.shape == (n_samples,
                                        self.history_length,
                                        self.action_dimension)

        assert action_future.shape == (n_samples,
                                       self.prediction_horizon - 1,
                                       self.action_dimension)

    def _check_prediction_outputs(self, observation_history, observation_prediction):
        n_samples = observation_history.shape[0]

        assert observation_prediction.shape == (n_samples,
                                                self.observation_dimension)


class DynamicsLearnerExample(DynamicsLearnerInterface):

    def name(self):
        return 'dynamics_learner_example'

    def _learn(self, training_inputs, training_targets):
        pass

    def _predict(self, single_input):
        return np.zeros((single_input.shape[0], self.observation_dimension))


if __name__ == '__main__':
    try:

        data = np.load('./Dataset/dataset_v01.npz')

        observation_sequences = np.concatenate((data['measured_angles'],
                                                data['measured_velocities'],
                                                data['measured_torques']), 2)

        action_sequences = data['constrained_torques']

        history_length = 10
        prediction_horizon = 100
        dynamics_learner = DynamicsLearnerExample(history_length, prediction_horizon)
        dynamics_learner.learn(observation_sequences, action_sequences)

        observation_prediction = dynamics_learner.predict(
                observation_sequences[:, :history_length],
                action_sequences[:, :history_length],
                action_sequences[:, history_length:history_length +
                prediction_horizon - 1])

        rms = np.linalg.norm(observation_sequences[:, history_length + prediction_horizon - 1] -
                             observation_prediction)
        print('rms: ', rms)

        ipdb.set_trace()

    except:
        traceback.print_exc(sys.stdout)
        _, _, tb = sys.exc_info()
        ipdb.post_mortem(tb)
