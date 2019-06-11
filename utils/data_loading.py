"""
basic helper functions to load the data set and get it in the right format
for different learning algorithms
"""
import os
import pickle
import numpy as np


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
    observations = np.concatenate((data['measured_angles'], data['measured_velocities'],
            data['measured_torques']), 2)
    actions = data['constrained_torques']
    return observations, actions


def concatenateActionsStates(history_actions, history_obs, future_actions):
    assert len(history_actions.shape) == 3
    assert len(history_obs.shape) == 3
    assert len(future_actions.shape) == 3
    assert history_actions.shape[:2] == history_obs.shape[:2]
    assert (history_actions.shape[0], history_actions.shape[2]) ==\
            (future_actions.shape[0], future_actions.shape[2])
    """
    concatenates observations and actions to form a single (row) vector. This
    function is intended to standardize the order observations and actions are
    merged to form the dynamics input. Note that it can handle multiple
    sequences at the same time.

    Parameters
    ----------

    history_actions:    np array with shape nsequences x history len x action dim
    history_obs:        np array with shape nsequences x history len x state dim
    future_actions:     np array with shape nsequences x prediction horizon - 1
                        x action dim.
    Returns
    -------

    joint_states_actions: np array with shape nsequences x (history len x action dim
                          + history len x state dim + (prediction horizon - 1)
                          * action dim).

    """
    joint_states_actions = [history_actions.reshape(
        (history_actions.shape[0], -1)), history_obs.reshape(
        (history_obs.shape[0], -1)), future_actions.reshape(
        (future_actions.shape[0], -1))]
    joint_states_actions = np.hstack(joint_states_actions)
    assert joint_states_actions.shape[0] == history_obs.shape[0]
    return joint_states_actions


def concatenateActionsStatesAverages(history_actions, history_obs, future_actions):
    assert len(history_actions.shape) == 3
    assert len(history_obs.shape) == 3
    assert len(future_actions.shape) == 3
    assert history_actions.shape[:2] == history_obs.shape[:2]
    assert (history_actions.shape[0], history_actions.shape[2]) ==\
            (future_actions.shape[0], future_actions.shape[2])
    """
    averages and concatenates observations and actions to form a single (row) vector. 
    See also concatenateActionsStates()

    Returns
    -------

    joint_states_actions: np array with shape 
                      nsequences x (input dim + state dim + (prediction horizon - 1) * input_dim)
    """
    if future_actions.shape[1]>0:
        joint_states_actions = [np.mean(history_actions, axis=1),
                                np.mean(history_obs, axis=1),
                                np.mean(future_actions, axis=1)]
    else:
        joint_states_actions = [np.mean(history_actions, axis=1),
                                np.mean(history_obs, axis=1)]
    joint_states_actions = np.hstack(joint_states_actions)
    assert joint_states_actions.shape[0] == history_obs.shape[0]
    return joint_states_actions


def unrollTrainingData(obs_seqs, actions_seqs, history_len, prediction_horizon,
        difference_learning, average=False):
    """
    Receives sequences of observations and actions and returns training targets
    and training inputs that will be used to learn the dynamics model.
    If average is True then the mean of the history and the mean of the future actions are used.

    Outputs
    -------
    targets:   np.array of shape training_instances x state dim

    inputs:    np-array of shape traininig_instances x input_dimension
               Note that input_dimension = (action dim+state dim)*history_len +
               (prediction_horizon - 1) x action dim
    """
    assert obs_seqs.shape[:2] == actions_seqs.shape[:2]
    inputs = []
    targets = []
    nrollouts, length, nstates = obs_seqs.shape
    for offset in range(history_len, length - prediction_horizon + 1):
        hist_obs = obs_seqs[:, offset - history_len:offset, :]
        hist_act = actions_seqs[:, offset - history_len:offset, :]
        future_act = actions_seqs[:,offset: offset + prediction_horizon - 1, :]
        output_obs = obs_seqs[:,offset + prediction_horizon - 1, :]
        if average:
            current_input = concatenateActionsStatesAverages(hist_act, hist_obs, future_act)
        else:
            current_input = concatenateActionsStates(hist_act, hist_obs, future_act)
        current_target = output_obs
        if difference_learning:
            current_target = current_target.copy() - hist_obs[:, -1, :]
        inputs.append(current_input)
        targets.append(current_target)
    return np.vstack(targets), np.vstack(inputs)


def unrollTrainingDataStream(obs_seqs, actions_seqs, history_len,
        prediction_horizon, difference_learning, shuffle=True, infinite=True):
    """
    Generator function that receives sequences of observations and actions and
    yields training pairs (target, input). Notice that the order of the pairs
    is shuffled by default. Moreover, the data iteration restarts from the
    beginning once the training pairs are exhausted if infinite=True (default).

    Outputs
    -------

    target: np array of size state dim.

    inputs: np array of size history_len * (action dim + state dim) +
            (prediction_horizon - 1) * action dim.
    """
    assert obs_seqs.shape[:2] == actions_seqs.shape[:2]
    nrollouts = obs_seqs.shape[0]
    ninstances = computeNumberOfTrainingPairs(obs_seqs, history_len,
            prediction_horizon)
    order = range(ninstances)
    if shuffle:
        order = np.random.permutation(ninstances)
    while True:
        for index in order:
            seq_id = index % nrollouts
            offset = index // nrollouts + history_len
            hist_obs = obs_seqs[seq_id, offset - history_len:offset, :]
            hist_act = actions_seqs[seq_id, offset - history_len:offset, :]
            future_act = actions_seqs[seq_id,
                    offset: offset + prediction_horizon - 1, :]
            output_obs = obs_seqs[seq_id, offset + prediction_horizon - 1, :]
            current_input = concatenateActionsStates(hist_act[np.newaxis, :, :],
                    hist_obs[np.newaxis, :, :], future_act[np.newaxis, :, :])
            current_target = output_obs
            if difference_learning:
                current_target = current_target.copy() - hist_obs[-1, :]
            yield (current_target.flatten(), current_input.flatten())
        if not infinite:
            break

def computeNumberOfTrainingPairs(obs_seqs, history_len, prediction_horizon):
    """
    Computes the number of different training pairs (target, input) for given
    sequences of observations and actions. Note that it also depends on the
    history length and prediction horizon.
    """
    nrollouts, length, _ = obs_seqs.shape
    valid_range_len = len(range(history_len, length - prediction_horizon + 1))
    ninstances = valid_range_len * nrollouts
    return ninstances
