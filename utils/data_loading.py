"""
basic helper functions to load the data set and get it in the right format
for different learning algorithms
"""
import os
import pickle


import numpy as np


from DL.dynamics_learner_interface.dynamics_learner_interface import DynamicsLearnerInterface


def loadRobotData(filename):
    """
    Loads Robot data at the given filename and returns it as a tuple of
    observations and actions.

    Returns
    ----------
    obs:        array of shape nRollouts x nStepsPerRollout x nStates
                containing the state trajectories of all rollouts
    actions:    array of shape nRollouts x nStepsPerRollout x nInputs
                containing the state trajectories of all rollouts
    """
    data = np.load(filename)
    observations = np.concatenate((data['measured_angles'],
            data['measured_torques']), 2)
    actions = data['constrained_torques']
    return observations, actions

def concatenateActionsStates(history_actions, history_obs, future_actions):
    """
    concatenates observations and actions to form a single vector. This function
    is intended to standardize the order observations and actions are merged to
    form the dynamics input.
    """
    return np.concatenate((history_actions.flatten(), history_obs.flatten(),
            future_actions.flatten()))

def unrollTrainingData(obs_seqs, actions_seqs, history_len, prediction_horizon,
        difference_learning=True):
    inputs = []
    targets = []
    for obs, act in zip(obs_seqs, actions_seqs):
        length = obs.shape[0]
        for offset in range(history_len, length - prediction_horizon + 1):
            hist_obs = obs[offset - history_len:offset]
            hist_act = act[offset - history_len:offset]
            future_act = act[offset: offset + prediction_horizon - 1]
            output_obs = obs[offset + prediction_horizon - 1]
            current_input = concatenateActionsStates(hist_act, hist_obs,
                    future_act)
            current_target = output_obs
            if difference_learning:
                current_target -= hist_obs[-1]
            inputs.append(current_input)
            targets.append(current_target)
    return np.vstack(targets), np.vstack(inputs)

def subsampleTrainingSet(inputs, targets, nsamples):
    """
    Returns a random subset of the training data set given as input.
    """
    assert inputs.shape[0] == targets.shape[0]
    n = inputs.shape[0]
    assert nsamples <= n
    permutation = np.random.permutation(n)[:nsamples]
    return inputs[permutation], targets[permutation]

class DataProcessor(DynamicsLearnerInterface):
    # Mix-in class to normalize inputs to train predict and (correspondingly) denormalize outputs of predict

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 1e-8
        self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_acts, self.std_acts = [None] * 6

    def learn(self, observation_sequences, action_sequences):
        # delta_states = np.zeros_like(observation_sequences)
        delta_states = observation_sequences[:, 1:, :] - observation_sequences[:, :-1, :]
        action_sequences = action_sequences[:, :-1, :]
        observation_sequences = observation_sequences[:,:-1,:]
        states = observation_sequences.reshape((observation_sequences.shape[0] * observation_sequences.shape[1], observation_sequences.shape[2]))
        delta_states = delta_states.reshape((delta_states.shape[0] * delta_states.shape[1], delta_states.shape[2]))
        actions = action_sequences.reshape((action_sequences.shape[0] * action_sequences.shape[1], action_sequences.shape[2]))
        #TODO: modify actions

        norm_states, norm_actions, norm_deltas = self.normalize(states, actions, delta_states, override_statistics=True)

        super().learn(norm_states, norm_actions, norm_deltas)

    def predict(self, observation_history, action_history, action_future):
        last_states = observation_history[-1, :].T
        states = last_states
        n_step_prediction = np.zeros((action_future.shape[0], observation_history.shape[1]))
        for i, action in enumerate(action_future):
            if states.ndim == 1:
                states = states[None, ...]
            if action.ndim == 1:
                action = action[None, ...]
            norm_states, norm_actions = self.normalize(states, action, override_statistics=False)
            norm_deltas = super().predict(norm_states, norm_actions)
            deltas = self.denormalize(norm_deltas)
            next_states = states + deltas
            n_step_prediction[i, :] = states + deltas
            states = next_states
        return n_step_prediction

        return states + deltas

    def normalize(self, states, actions, delta_states=None, override_statistics=False):
        if override_statistics:
            self.mean_states = np.mean(states, axis=0)
            self.std_states = np.std(states, axis=0)
            self.mean_acts = np.mean(actions, axis=0)
            self.std_acts = np.std(actions, axis=0)
        normalized_states = (states - self.mean_states) / (self.std_states + self.eps)
        normalized_actions = (actions - self.mean_acts) / (self.std_acts + self.eps)

        if delta_states is not None:
            if override_statistics:
                self.mean_deltas = np.mean(delta_states, axis=0)
                self.std_deltas = np.std(delta_states, axis=0)
            normalized_deltas = (delta_states - self.mean_deltas) / (self.std_deltas + self.eps)
            return normalized_states, normalized_actions, normalized_deltas
        else:
            return normalized_states, normalized_actions

    def denormalize(self, deltas):
        return deltas * self.std_deltas + self.mean_deltas

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        normalization_stats = [self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_acts, self.std_acts]
        with open(os.path.join(model_dir ,"normalization_stats.pickle"), 'wb') as handle:
            pickle.dump(normalization_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        super().save(model_dir)

    def load(self, model_dir):
         with open(os.path.join(model_dir ,"normalization_stats.pickle"), 'rb') as handle:
             normalization_stats = pickle.load(handle)
         self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_acts, self.std_acts = normalization_stats
         super().load(model_dir)

def connected_shuffle(list_arrays):
    random_state = np.random.RandomState(0)
    n_samples = list_arrays[0].shape[0]
    numels = np.array([a.shape[0] for a in list_arrays])
    if not np.all(numels == n_samples):
        raise ValueError('Different number of elements along axis 0', numels, n_samples)
    shuffling_indices = random_state.permutation(n_samples)
    list_arrays = [a[shuffling_indices] for a in list_arrays]
    return list_arrays
