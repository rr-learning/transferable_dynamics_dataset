"""
Evaluation funcionality. Note that it can also be called as a script.
"""
import ipdb
import argparse
import numpy as np
# from DL.methods.pilco_dynamics_learner import PilcoDynamicsLearner
from DL.dynamics_learner_interface.dynamics_learner_interface import DynamicsLearnerExample

from DL.utils.data_loading import loadRobotData


def evaluate(dynamics_learner, observation_sequences, action_sequences, dataset_name):
    possible_history_lengths = [1, 10]
    possible_prediction_horizons = [1, 10, 100, 1000]
    assert dynamics_learner.history_length in possible_history_lengths
    assert dynamics_learner.prediction_horizon in possible_prediction_horizons

    history_length = dynamics_learner.history_length
    if dynamics_learner.prediction_horizon == 1:
        prediction_horizons = possible_prediction_horizons
    else:
        prediction_horizons = [dynamics_learner.prediction_horizon]

    for prediction_horizon in prediction_horizons:
        T = range(possible_history_lengths[-1] - 1,
                  observation_sequences.shape[1] - possible_prediction_horizons[-1])
        errors = np.empty((observation_sequences.shape[0],
                           len(T),
                           observation_sequences.shape[2]))
        for i in xrange(len(T)):
            t = T[i]
            observation_history = observation_sequences[:, t + 1 - history_length: t + 1]
            action_history = action_sequences[:, t + 1 - history_length: t + 1]
            action_future = action_sequences[:, t + 1: t + prediction_horizon]
            observation_prediction = dynamics_learner.predict(observation_history=observation_history,
                                                              action_history=action_history,
                                                              action_future=action_future)
            true_observation = observation_sequences[:, t + prediction_horizon]
            errors[:, i] = observation_prediction - true_observation

        filename = dynamics_learner.name() + \
                   '__history_' + str(history_length) + \
                   '__training_horizon_' + str(dynamics_learner.prediction_horizon) + \
                   '__evaluation_horizon_' + str(prediction_horizon) + '__' + dataset_name

        np.save(filename, errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True, help="<Required> filename of"
            " the input robot data")
    parser.add_argument("--method", required=True, help="<Required> Name of the"
            " method that will be tested", choices=['example', 'pilco'])
    args = parser.parse_args()

    observations, actions = loadRobotData(args.data_filename)

    if args.method == 'example':
        dynamics_learner = DynamicsLearnerExample(1, 10)
        dynamics_learner.learn(observations, actions)
        evaluate(dynamics_learner, observations, actions, 'some_identifier_for_dataset')
    elif args.method == 'pilco':
        pass
        #
        # # TODO: Add the following hyperparameters to the Settings directory
        # # and load it as a commmand line argument.
        # ninducing = 10
        # ntraining = 10
        # dynamics_learner = PilcoDynamicsLearner(ninducing, ntraining)
        # dynamics_learner.load('some_filename')
        # evaluate(dynamics_learner, observations, actions, args.data_filename)
    else:
        raise NotImplementedError
