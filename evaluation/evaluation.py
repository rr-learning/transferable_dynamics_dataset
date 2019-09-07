"""
Evaluation functionality. Note that it can also be called as a script.
"""
import os
import json
import argparse
import numpy as np
import time

import sys
import ipdb

from DL.dynamics_learner_interface.dynamics_learner_interface import DynamicsLearnerExample
from DL.utils import Standardizer
from DL.utils.data_loading import loadRobotData


# hack because if have to use python 2.7 and it does not seem to have perf_counter
if sys.version_info[0] < 3:
    def return_0():
        return 0
    time.perf_counter = return_0


def evaluate(dynamics_learner, observation_sequences, action_sequences,
        test_dataset_name, verbose=False):
    possible_history_lengths = [1, 10]
    possible_prediction_horizons = [1, 10, 100, 1000]
    assert dynamics_learner.history_length in possible_history_lengths
    assert dynamics_learner.prediction_horizon in possible_prediction_horizons

    history_length = dynamics_learner.history_length

    # Only evaluating in the prediction horizon that a model was trained on.
    prediction_horizons = [dynamics_learner.prediction_horizon]

    output_errors = {}
    for prediction_horizon in prediction_horizons:
        T = range(possible_history_lengths[-1] - 1,
                  observation_sequences.shape[1] - possible_prediction_horizons[-1])
        errors = np.empty((observation_sequences.shape[0],
                           len(T),
                           observation_sequences.shape[2]))
        times = []
        init_pred_time = time.perf_counter()
        for i in range(len(T)):
            t = T[i]
            observation_history = observation_sequences[:, t + 1 - history_length: t + 1]
            action_history = action_sequences[:, t + 1 - history_length: t + 1]
            action_future = action_sequences[:, t + 1: t + prediction_horizon]
            start_time = time.perf_counter()
            observation_prediction = dynamics_learner.predict(
                    observation_history=observation_history,
                    action_history=action_history,
                    action_future=action_future)
            times.append(time.perf_counter() - start_time)
            true_observation = observation_sequences[:, t + prediction_horizon]
            errors[:, i] = true_observation - observation_prediction
        end_pred_time = time.perf_counter()
        if verbose:
            times = np.array(times)
            print('Number of predict calls {}'.format(len(times)))
            print('Elapsed time for predict call {}: {} +- {}'.format(
                    test_dataset_name, np.mean(times), np.std(times)))
            print('Total prediction time {}'.format(
                    end_pred_time - init_pred_time))

        errors_key = test_dataset_name + '__history_' + str(history_length) + \
                '__training_horizon_' + \
                str(dynamics_learner.prediction_horizon) + \
                '__evaluation_horizon_' + str(prediction_horizon)
        output_errors[errors_key] = errors

    # Right now we only test on the same setup used for training.
    # Therefore, there must be only one entry in the dictionary.
    errors_to_return = list(output_errors.values())
    assert len(errors_to_return) == 1
    return errors_to_return[0]

def get_evaluation_errors(all_errors):
    evaluation_errors = {}
    evaluation_errors['angle'] = all_errors[:,:, :3]
    evaluation_errors['velocity'] = all_errors[:,:, 3:6]
    evaluation_errors['torque'] = all_errors[:,:, 6:9]

    for key in evaluation_errors.keys():
        norms = np.linalg.norm(evaluation_errors[key], axis=-1, ord=1)
        norms = np.sum(norms, axis=1)

        # norms = norms / evaluation_errors[key].size * norms.size
        evaluation_errors[key] = norms.flatten()

    return evaluation_errors



def get_angle_errors(errors):
    """
    Takes error vectors computed over full state predictions and picks the
    dimensions corresponding to angle predictions. Notice that it is assumed
    that the first three dimenions contain angle errors.
    """
    return errors[:,:, :3]

def compute_RMSE_from_errors(errors):
    """
    Computes the RMSE from the error vectors. Notice that it weights equally
    all dimensions.
    """
    nseq, length, state_dim = errors.shape
    errors = errors.reshape((-1, state_dim))
    squared_errors = np.sum(errors * errors, axis=1)
    return np.sqrt(np.mean(squared_errors))


def run(parser):
    parser.add_argument("--training_data", required=True,
                        help="<Required> filename of the input robot training data")
    parser.add_argument("--trained_model",
                        help="filename of a trained model. If specified the model won't be"
                             " trained")
    parser.add_argument("--settings",
                        help="filename where the model settings are stored")
    parser.add_argument("--validation_data",
                        help="filename of the input robot validation data")
    parser.add_argument("--iid_test_data",
                        help="filename of the input robot iid testing data")
    parser.add_argument("--transfer_test_data", nargs='+',
                        help="filename of the input robot transfer testing data")
    parser.add_argument("--method", required=True,
                        help="<Required> Name of the method that will be tested")
    parser.add_argument("--history_length", type=int, default=1)
    parser.add_argument("--prediction_horizon", type=int, default=1)
    parser.add_argument("--output_errors",
                        help="<Required> filename where the computed errors will be saved")
    parser.add_argument("--output_model",
                        help="filename where the trained model will be saved if a trained"
                             " model was not already provided in the command line.")
    parser.add_argument("--averaging", dest='averaging', action='store_true')
    parser.add_argument("--no-averaging", dest='averaging',
                        action='store_false')
    parser.add_argument("--streaming", dest='streaming', action='store_true')
    parser.add_argument("--no-streaming", dest='streaming',
                        action='store_false')
    parser.add_argument("--verbose", action='store_true')
    parser.set_defaults(averaging=False)
    parser.set_defaults(streaming=False)
    arguments = parser.parse_args()
    history_length = arguments.history_length
    prediction_horizon = arguments.prediction_horizon
    settings = None
    if arguments.settings:
        with open(arguments.settings, 'r') as f:
            settings = json.load(f)
    dynamics_learner = None
    if arguments.method == 'example':
        dynamics_learner = DynamicsLearnerExample(history_length,
                                                  prediction_horizon,
                                                  averaging=arguments.averaging,
                                                  streaming=arguments.streaming)
    elif arguments.method == 'pilco_ninducing_500_ntraining_50000':
        from DL.methods.pilco_dynamics_learner import PilcoDynamicsLearner

        ninducing = 500
        ntraining = 50000
        dynamics_learner = PilcoDynamicsLearner(history_length,
                                                prediction_horizon, ninducing,
                                                ntraining,
                                                averaging=arguments.averaging,
                                                streaming=arguments.streaming)
    elif arguments.method == 'SVGPR':
        from DL.methods.SVGPR import SVGPR

        ninducing = 1000
        minibatch_size = 1000
        epochs = 40
        dynamics_learner = SVGPR(history_length,
                                 prediction_horizon, ninducing, minibatch_size,
                                 epochs=epochs, averaging=arguments.averaging,
                                 streaming=arguments.streaming)
    elif arguments.method == 'linear_model_ls':
        from DL.methods.linear_regression_ls import LinearModel

        dynamics_learner = LinearModel(history_length, prediction_horizon,
                                       averaging=arguments.averaging,
                                       streaming=arguments.streaming)
    elif arguments.method == 'linear_model_sgd':
        from DL.methods.linear_regression_sgd import LinearModelSGD

        dynamics_learner = LinearModelSGD(history_length, prediction_horizon,
                                          difference_learning=True,
                                          averaging=arguments.averaging,
                                          streaming=arguments.streaming,
                                          settings=settings)
    elif arguments.method == 'BNN':
        from DL.methods.BNN import BNNLearner

        dynamics_learner = BNNLearner(history_length, prediction_horizon,
                                      averaging=arguments.averaging,
                                      streaming=arguments.streaming)
    elif arguments.method == 'NN':
        from DL.methods.nn_dynamics_learner import NNDynamicsLearner
        settings_file = "./Settings/nn_prediction_horizon_{0}_history_length_{1}.json".format(
            prediction_horizon, history_length)
        exists = os.path.isfile(settings_file)
        if exists:
            with open(settings_file, 'r') as f:
                params = json.load(f)
            dynamics_learner = NNDynamicsLearner(history_length=history_length,
                                                 prediction_horizon=prediction_horizon,
                                                 model_arch_params=params[
                                                     "model_arch_params"],
                                                 model_train_params=params[
                                                     "model_train_params"],
                                                 mode=params['mode'],
                                                 averaging=arguments.averaging,
                                                 streaming=arguments.streaming)
    elif arguments.method == 'EQL':
        from DL.methods.eql_dynamics_learner import EQL
        settings_file = "./Settings/eql_prediction_horizon_{0}_history_length_{1}.json".format(
            prediction_horizon, history_length)
        exists = os.path.isfile(settings_file)
        if exists:
            with open(settings_file, 'r') as f:
                params = json.load(f)
            dynamics_learner = EQL(history_length=history_length,
                                   prediction_horizon=prediction_horizon,
                                   model_arch_params=params[
                                       "model_arch_params"],
                                   model_train_params=params[
                                       "model_train_params"],
                                   optional_params=params["optional_params"],
                                   averaging=arguments.averaging,
                                   streaming=arguments.streaming)
    elif arguments.method == 'Eureqa':
        from DL.methods.eureqa_dynamics_learner import Eureqa
        dynamics_learner = Eureqa(history_length=history_length,
                                  prediction_horizon=prediction_horizon,
                                  averaging=arguments.averaging,
                                  streaming=arguments.streaming)
    elif arguments.method == 'lwpr':
        from DL.methods.LWPR import lwpr_dyn_model
        dynamics_learner = lwpr_dyn_model(history_length, prediction_horizon,
                                          difference_learning=True,
                                          averaging=arguments.averaging,
                                          streaming=arguments.streaming,
                                          settings=settings)
    elif arguments.method == 'system_id':
        from DL.methods.system_id import SystemId
        dynamics_learner = SystemId(history_length=arguments.history_length,
                                    prediction_horizon=arguments.prediction_horizon,
                                    settings=settings)
    assert dynamics_learner, "Make sure the method is implemented."
    training_observations, training_actions = loadRobotData(
        arguments.training_data)
    if arguments.trained_model:
        dynamics_learner.load_normalization_stats(training_observations,
                                                  training_actions)
        dynamics_learner.load(arguments.trained_model)
    else:
        initial_time = time.perf_counter()
        dynamics_learner.learn(training_observations, training_actions)
        if arguments.verbose:
            print('Training time {} s'.format(
                time.perf_counter() - initial_time))
        if arguments.output_model:
            dynamics_learner.save(arguments.output_model)

    datasets = {}
    if arguments.transfer_test_data:
        for i, dataset_path in enumerate(arguments.transfer_test_data):
            datasets['transfer_test_data_{}'.format(i + 1)] = dataset_path
    for dataset in ['training_data', 'iid_test_data', 'validation_data']:
        dataset_path = getattr(arguments, dataset)
        if dataset_path:
            datasets[dataset] = dataset_path

    # Maps each data set to its corresponding error file.
    set_to_errors = {}
    for dataset in sorted(datasets.keys()):
        dataset_path = datasets[dataset]
        testing_observations, testing_actions = loadRobotData(dataset_path)
        print("evaluating on ", dataset_path, testing_observations.shape)

        errors = evaluate(dynamics_learner, testing_observations,
                          testing_actions, dataset, verbose=arguments.verbose)
        set_to_errors[dataset] = errors
        print("{} error:".format(dataset))
        angle_errors = get_angle_errors(errors)
        print(compute_RMSE_from_errors(angle_errors))
    if arguments.output_errors:
        np.savez(arguments.output_errors, **set_to_errors)


if __name__ == "__main__":
    # run(parser=argparse.ArgumentParser(description=__doc__))


    import ipdb
    import traceback
    try:
        run(parser=argparse.ArgumentParser(description=__doc__))
    except:
        traceback.print_exc(sys.stdout)
        _, _, tb = sys.exc_info()
        ipdb.post_mortem(tb)

