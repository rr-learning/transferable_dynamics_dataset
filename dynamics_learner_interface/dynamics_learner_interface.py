import numpy as np
import sys
import ipdb
import traceback
from collections import defaultdict
from DL.utils import unrollTrainingData, concatenateActionsStates, \
        Standardizer, concatenateActionsStatesAverages, \
        unrollTrainingDataStream, computeNumberOfTrainingPairs


class DynamicsLearnerInterface(object):

    def __init__(self, history_length, prediction_horizon,
            difference_learning=True, averaging=False, streaming=False):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.observation_dimension = 9
        self.action_dimension = 3
        self.difference_learning = difference_learning
        self.averaging = averaging
        self.streaming = streaming

    # do not override this function!
    def learn(self, observation_sequences, action_sequences):
        self._check_learning_inputs(observation_sequences, action_sequences)

        self.load_normalization_stats(observation_sequences, action_sequences)

        if self.streaming:
            training_data_stream = unrollTrainingDataStream(
                    observation_sequences, action_sequences,
                    self.history_length, self.prediction_horizon,
                    self.difference_learning, infinite=True)
            ntraining_pairs = computeNumberOfTrainingPairs(
                    observation_sequences, self.history_length,
                    self.prediction_horizon)
            normalized_data_stream = self._standardize_data_stream(
                    training_data_stream)
            self._learn_from_stream(normalized_data_stream, ntraining_pairs)
        else:
            targets, inputs = unrollTrainingData(observation_sequences,
                    action_sequences, self.history_length,
                    self.prediction_horizon, self.difference_learning,
                    self.averaging)

            # Whitening the inputs.
            std_targets = self.targets_standardizer.standardize(targets)
            std_inputs = self.inputs_standardizer.standardize(inputs)
            self._learn(std_inputs, std_targets)

    def _standardize_data_stream(self, data_stream):
        for training_target, training_input in data_stream:
            yield (self.targets_standardizer.standardize(training_target),
                    self.inputs_standardizer.standardize(training_input))

    def _training_inputs_data_stream(self, data_stream):
        for _, training_input in data_stream:
            yield training_input

    def _training_targets_data_stream(self, data_stream):
        for training_target, _ in data_stream:
            yield training_target

    # do not override this function!
    def _preprocess_and_predict(self, observation_history, action_history, action_future=None):
        if action_future is None:
            assert self.prediction_horizon == 1
            action_future = np.empty((observation_history.shape[0],
                                      0,
                                      self.action_dimension))

        self._check_prediction_inputs(observation_history, action_history, action_future)

        # Making a single input from all the input parameters.
        if self.averaging:
            dynamics_inputs = concatenateActionsStatesAverages(action_history,
                    observation_history, action_future)
        else:
            dynamics_inputs = concatenateActionsStates(action_history,
                    observation_history, action_future)

        # Whitening the input.
        whitened_input = self.inputs_standardizer.standardize(dynamics_inputs)

        whitened_predictions = self._predict(whitened_input)

        # Dewhitening the output
        dewhitened_predictions = self.targets_standardizer.unstandardize(
                whitened_predictions)

        if self.difference_learning:
            dewhitened_predictions += observation_history[:, -1, :]

        self._check_prediction_outputs(observation_history,
                dewhitened_predictions)
        return dewhitened_predictions

    # do not override this function!
    def predict(self, observation_history, action_history, action_future=None):
        if action_future is None:
            assert self.prediction_horizon == 1
            action_future = np.empty((observation_history.shape[0],
                                      0,
                                      self.action_dimension))

        if self.prediction_horizon == action_future.shape[1] + 1:
            return self._preprocess_and_predict(observation_history, action_history, action_future)

        assert self.prediction_horizon == 1

        observation_history_t = observation_history
        action_history_t = action_history
        predicted_observation = self._preprocess_and_predict(observation_history_t, action_history_t)

        for t in range(action_future.shape[1]):
            predicted_observation = np.expand_dims(predicted_observation, axis=1)
            observation_history_t = np.append(observation_history_t[:, 1:],
                    predicted_observation, axis=1)
            action_history_t = np.append(action_history_t[:, 1:],
                    action_future[:, t:t + 1], axis=1)
            predicted_observation = self._preprocess_and_predict(
                    observation_history_t, action_history_t)

            assert (action_history_t[:, :-(t + 1)] == action_history[:, t + 1:]).all()
            assert (observation_history_t[:, :-(t + 1)] == observation_history[:, t + 1:]).all()
            assert (action_history_t[:, -1] == action_future[:, t]).all()

        return predicted_observation

    # override this function
    def name(self):
        raise NotImplementedError

    # override this function
    def _learn(self, training_inputs, training_targets):
        """
        Learns from the entire batch of training pairs.

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
    def _learn_from_stream(self, training_datastream, datastream_size):
        """
        Learns from a data stream which iterates over the training set. This
        way there is no need to have the whole data set in memory.

        Parameters
        ----------
        training_datastream: Python generator that yields (target, input) pairs
                             of the training data set.

        datastream_size:     Number of training pairs in training_datastream.
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

    def load_normalization_stats(self, observation_sequences, action_sequences):
        """
        Loads the normalization statistics from the input data.
        """
        self._check_learning_inputs(observation_sequences, action_sequences)
        if not self.streaming:
            targets, inputs = unrollTrainingData(observation_sequences,
                    action_sequences, self.history_length,
                    self.prediction_horizon, self.difference_learning,
                    self.averaging)
        else:
            targets = self._training_targets_data_stream(
                    unrollTrainingDataStream(
                    observation_sequences, action_sequences,
                    self.history_length, self.prediction_horizon,
                    self.difference_learning, infinite=False))
            inputs = self._training_inputs_data_stream(
                    unrollTrainingDataStream(
                    observation_sequences, action_sequences,
                    self.history_length, self.prediction_horizon,
                    self.difference_learning, infinite=False))

        # Loading the standardizers.
        self.targets_standardizer = Standardizer(targets)
        self.inputs_standardizer = Standardizer(inputs)

    # Override this function.
    def load(self, model_filename):
        raise NotImplementedError

    # Override this function.
    def save(self, model_filename):
        raise NotImplementedError


class DynamicsLearnerExample(DynamicsLearnerInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self):
        return 'dynamics_learner_example'

    def _learn(self, training_inputs, training_targets):
        pass

    def _learn_from_stream(self, training_data_stream):
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
        dynamics_learner = DynamicsLearnerExample(history_length,
                prediction_horizon, streaming=True)
        dynamics_learner.learn(observation_sequences, action_sequences)

        hist_obs = observation_sequences[:, :history_length].copy()
        hist_act = action_sequences[:, :history_length].copy()
        fut_act = action_sequences[:, history_length:history_length +
                prediction_horizon - 1].copy()
        observation_prediction = dynamics_learner.predict(hist_obs, hist_act,
                fut_act)
        rms = np.linalg.norm(observation_sequences[:, history_length + prediction_horizon - 1] -
                             observation_prediction)

        # Asserting that the inputs to the predict method were left unchanged.
        assert np.array_equal(hist_obs,
                observation_sequences[:, :history_length])
        assert np.array_equal(hist_act,
                action_sequences[:, :history_length])
        assert np.array_equal(fut_act, action_sequences[:,
                history_length:history_length + prediction_horizon - 1])
        print('rms: ', rms)

        ipdb.set_trace()

    except:
        traceback.print_exc(sys.stdout)
        _, _, tb = sys.exc_info()
        ipdb.post_mortem(tb)
