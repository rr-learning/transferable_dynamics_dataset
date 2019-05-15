import numpy as np
import sys
import ipdb
import traceback


class DynamicsLearnerInterface(object):

    def __init__(self, history_length, prediction_horizon):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.observation_dimension = 9
        self.action_dimension = 3

    # do not override this function!
    def learn(self, observation_sequences, action_sequences):
        self._check_learning_inputs(observation_sequences, action_sequences)
        self._learn(observation_sequences, action_sequences)

    # do not override this function!
    def predict(self, observation_history, action_history, action_future=None):
        self._check_prediction_inputs(observation_history, action_history, action_future)
        observation_prediction = self._predict(observation_history, action_history, action_future)
        self._check_prediction_outputs(observation_history, observation_prediction)
        return observation_prediction

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
            observation_history_t = np.append(observation_history_t[:, 1:], predicted_observation, axis=1)
            action_history_t = np.append(action_history_t[:, 1:], action_future[:, t:t + 1], axis=1)
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
    def _learn(self, observation_sequences, action_sequences):
        """
        Parameters
        ----------
        observation_sequences:  np-array of shape nSequences x nStepsPerRollout x observation_dimension
                                past state observations
        action_sequences:       np-array of shape nSequences x nStepsPerRollout x action_dimension
                                actions taken at the corresponding time points.
        """
        raise NotImplementedError

    # override this function
    def _predict(self, observation_history, action_history, action_future):
        """
        Parameters
        ----------
        observation_history:    np-array of shape n_samples x
                                self.history_length x observation_dimension
        action_history:         np-array of shape n_samples x
                                self.history_length x action_dimension
        action_future:          np-array of shape n_samples x
                                self.prediction_horizon - 1 x action_dimension
                                actions to be applied to the system. The first
                                action is the action applied one time step after
                                the last action of the corresponding
                                "action_history".
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

        if self.prediction_horizon == 1:
            assert action_future is None
        else:
            assert action_future.shape == (n_samples,
                                           self.prediction_horizon - 1,
                                           self.action_dimension)

    def _check_prediction_outputs(self, observation_history, observation_prediction):
        n_samples = observation_history.shape[0]

        assert observation_prediction.shape == (n_samples,
                                                self.observation_dimension)


class DynamicsLearnerExample(DynamicsLearnerInterface):

    def _learn(self, observation_sequences, action_sequences):
        pass

    def _predict(self, observation_history, action_history, action_future):
        return observation_history[:, -1, :]


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

        observation_prediction = dynamics_learner.predict(observation_sequences[:, :history_length],
                                                          action_sequences[:, :history_length],
                                                          action_sequences[:, history_length:history_length
                                                                                             + prediction_horizon - 1])

        rms = np.linalg.norm(observation_sequences[:, history_length + prediction_horizon - 1] -
                             observation_prediction)
        print('rms: ', rms)

        ipdb.set_trace()

    except:
        traceback.print_exc(sys.stdout)
        _, _, tb = sys.exc_info()
        ipdb.post_mortem(tb)
