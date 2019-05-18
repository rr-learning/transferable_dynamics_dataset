"""
Evaluation funcionality. Note that it can also be called as a script.
"""
import os
import ipdb
import argparse
import numpy as np
from DL.dynamics_learner_interface.dynamics_learner_interface import DynamicsLearnerExample
from DL.methods import LinearModelSGD, PilcoDynamicsLearner
from DL.utils.data_loading import loadRobotData


def evaluate(dynamics_learner, observation_sequences, action_sequences,
        test_dataset_name):
    possible_history_lengths = [1, 10]
    possible_prediction_horizons = [1, 10, 100, 1000]
    assert dynamics_learner.history_length in possible_history_lengths
    assert dynamics_learner.prediction_horizon in possible_prediction_horizons

    history_length = dynamics_learner.history_length
    if dynamics_learner.prediction_horizon == 1:
        prediction_horizons = possible_prediction_horizons
    else:
        prediction_horizons = [dynamics_learner.prediction_horizon]

    output_errors = {}
    for prediction_horizon in prediction_horizons:
        T = range(possible_history_lengths[-1] - 1,
                  observation_sequences.shape[1] - possible_prediction_horizons[-1])
        errors = np.empty((observation_sequences.shape[0],
                           len(T),
                           observation_sequences.shape[2]))
        for i in range(len(T)):
            t = T[i]
            observation_history = observation_sequences[:, t + 1 - history_length: t + 1]
            action_history = action_sequences[:, t + 1 - history_length: t + 1]
            action_future = action_sequences[:, t + 1: t + prediction_horizon]
            observation_prediction = dynamics_learner.predict(observation_history=observation_history,
                                                              action_history=action_history,
                                                              action_future=action_future)
            true_observation = observation_sequences[:, t + prediction_horizon]
            errors[:, i] = observation_prediction - true_observation

        errors_key = dynamics_learner.name() + '__history_' + \
                str(history_length) + '__training_horizon_' + \
                str(dynamics_learner.prediction_horizon) + \
                '__evaluation_horizon_' + str(prediction_horizon) + '__' + \
                test_dataset_name
        output_errors[errors_key] = errors
    return output_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_data", required=True,
            help="<Required> filename of the input robot trainig data")
    parser.add_argument("--testing_data", required=True,
            help="<Required> filename of the input robot testing data")
    parser.add_argument("--method", required=True,
            help="<Required> Name of the method that will be tested")
    parser.add_argument("--output", required=True,
            help="<Required> filename where the computed errors will be saved")
    args = parser.parse_args()
    dynamics_learner = None
    if args.method == 'example':
        dynamics_learner = DynamicsLearnerExample(1, 1)
    elif args.method == 'pilco_ninducing_500_ntraining_50000':
        ninducing = 500
        ntraining = 50000
        history_length = 1
        prediction_horizon = 1
        dynamics_learner = PilcoDynamicsLearner(history_length,
                prediction_horizon, ninducing, ntraining)
        print("Training Done")
    elif args.method == 'linear_model_sgd':
        dynamics_learner = LinearModelSGD(1, 1)
    assert dynamics_learner, "Make sure the method is implemented."
    training_observations, training_actions = loadRobotData(args.training_data)
    testing_observations, testing_actions = loadRobotData(args.testing_data)
    dynamics_learner.learn(training_observations, training_actions)
    errors = evaluate(dynamics_learner, testing_observations,
            testing_actions, args.testing_data)
    np.savez(args.output, **errors)

