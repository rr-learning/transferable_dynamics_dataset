"""
basic helper functions to load the data set and get it in the right format
for different learning algorithms
"""

import numpy as np

def loadTrainingData(path):
    """
    Loads the training data at the given path.
    Defalut will return a small example dataset of 10*5000 points
    
    Returns
    ----------
    obs:        array of shape nRollouts x nStepsPerRollout x nStates
                containing the state trajectories of all rollouts
    actions:    array of shape nRollouts x nStepsPerRollout x nInputs
                containing the state trajectories of all rollouts
    """
    data = np.load(path)
    observations = np.concatenate((data['measured_angles'],
        data['measured_velocities'], data['measured_torques']), 2)
    actions = data['constrained_torques']
    return observations, actions

def unrollForDifferenceTraining(obs, actions, offset=0):
    """
    Returns vectors ready for training of a difference equation model. A
    difference equation model should predict targets[i, :] = model(inputs[i, :])
    
    Parameters
    ----------
    obs:        array of shape nRollouts x nStepsPerRollout x nStates
                containing the state trajectories of all rollouts
    actions:    array of shape nRollouts x nStepsPerRollout x nInputs
                containing the state trajectories of all rollouts
    offset:     int denoting the index (inclusive) from which the rollouts
                will be taken into account; i.e, to discard the first part.
                containing the state trajectories of all rollouts
    
    Returns
    ----------
    targets:    array of shape nRollouts*(nStepsPerRollout-1) x nStates
                state increment targets for training
    inputs:     array of shape (nRollouts*nStepsPerRollout-1) x (nStates+nInputs)
                states and actions concatenated.
    """
    obs = obs[:, offset:, :]
    actions = actions[:, offset:, :]
    targets = obs[:, 1:, :] - obs[:, :-1, :]
    targets = np.reshape(targets, [targets.shape[0]*targets.shape[1], targets.shape[2]])

    actionInputs = np.reshape(actions[:, :-1, :], [targets.shape[0], -1])
    unrolledStates = np.reshape(obs[:, :-1, :], [targets.shape[0], targets.shape[1]])
    
    inputs = np.concatenate([actionInputs, unrolledStates], axis=1)

    return targets, inputs
