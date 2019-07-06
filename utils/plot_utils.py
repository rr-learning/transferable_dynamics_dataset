import pandas as pd
import numpy as np
import re
import os
from DL.evaluation.evaluation import get_angle_errors, compute_RMSE_from_errors
import re


def get_path_to_run(num_layers, num_units, lr, reg,  path_to_ho="/is/cluster/azadaianchuk/hyperparameter_optimization/dynamics_learning/NN/split_5/prediction_horizon_100_history_length_1_epochs_40/"):
    # path_to_ho="/is/cluster/azadaianchuk/hyperparameter_optimization/dynamics_learning/NN/split_5/prediction_horizon_100_history_length_1_epochs_40/"
    jobs_info = pd.read_csv(os.path.join(path_to_ho, "job_info.csv"))
    jobs_info["model_train_params"]
    run_id = None
    for arch, train, id in zip(jobs_info["model_arch_params"], jobs_info["model_train_params"], jobs_info["id"]):
        arch = eval(arch)
        train = eval(train)
        if arch["num_layers"] == num_layers and arch["num_units"] == num_units and train["learning_rate"] == lr and train["l2_reg"] == reg:
            run_id = id
    if run_id is not None:
        return os.path.join(path_to_ho, "{}_".format(run_id), "errors.npz")
    else:
        print("No such run in given folder")


def get_diego_index(prediction_horizon,
                    history_length,
                    averaging):
    prediction_horizons = [1, 10, 100, 1000]
    history_lengths = [1, 10]

    count = 0
    for current_prediction_horizon in prediction_horizons:
        for current_history_length in history_lengths:
            for current_averaging in [True, False]:
                if prediction_horizon == current_prediction_horizon and \
                        history_length == current_history_length and \
                        averaging == current_averaging:
                    return count

                count += 1
    return np.nan

def path_to_error_file(method_name,
                       experiment_name,
                       prediction_horizon,
                       history_length):
    if experiment_name == 'sine':
        path_to_results = "/agbs/dynlearning/Errors/new_datasets/SinePD/"
    else:
        raise NotImplementedError

    if method_name == 'avg-NN':
        error_file_name = "NN/averaging_prediction_horizon_{}_history_length" \
                          "_{}_epochs_40/errors.npz".format(prediction_horizon,
                                                            history_length)
    elif method_name == 'NN':
        error_file_name = "NN/prediction_horizon_{}_history_length" \
                          "_{}_epochs_40/errors.npz".format(prediction_horizon,
                                                            history_length)
    elif method_name == 'avg-EQL':
        error_file_name = "EQL/averaging_prediction_horizon_{}_history_length" \
                          "_{}_epochs_20/errors.npz".format(prediction_horizon,
                                                            history_length)
    elif method_name == 'delta 0':
        error_file_name = 'delta_0/errors_sine_pd_delta_0_{:03d}.npz'.format(
            get_diego_index(prediction_horizon=prediction_horizon,
                            history_length=history_length,
                            averaging=False))
    elif method_name == 'svgpr':
        error_file_name = 'svgpr/errors_sine_pd_svgpr_{:03d}.npz'.format(
            get_diego_index(prediction_horizon=prediction_horizon,
                            history_length=history_length,
                            averaging=False))
    elif method_name == 'avg-svgpr':
        error_file_name = 'svgpr/errors_sine_pd_svgpr_{:03d}.npz'.format(
            get_diego_index(prediction_horizon=prediction_horizon,
                            history_length=history_length,
                            averaging=True))
    elif method_name == 'linear':
        error_file_name = 'linear_model_learning_rate_0.0001/errors_sine' \
                          '_pd_linear_model_{:03d}.npz'.format(
            get_diego_index(prediction_horizon=prediction_horizon,
                            history_length=history_length,
                            averaging=False))
    elif method_name == 'avg-linear':
        error_file_name = 'linear_model_learning_rate_0.0001/errors_sine' \
                          '_pd_linear_model_{:03d}.npz'.format(
            get_diego_index(prediction_horizon=prediction_horizon,
                            history_length=history_length,
                            averaging=True))
    elif method_name in ['sys id cad', 'sys id ls', 'sys id ls-lmi']:
        identification_method = method_name[7:]
        error_file_name = 'system_id/system_id__' + experiment_name + \
                          '__horizon_' + str(prediction_horizon).zfill(4) + \
                          '__history_' + str(history_length).zfill(2) + \
                          '__identification_method_' + identification_method + \
                          '.npz'
    elif bool(re.compile("NN_layers_[0-9]_width_[0-9]+_lr_0.0001_reg_0.0001").match(method_name)):
        method_name = "NN_layers__3_width__64_lr_0.0001_reg_0.0001"
        pattern2 = re.compile("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?")
        (num_layers, num_units, lr, reg) = [float(name) for name in pattern2.findall(method_name)]
        return get_path_to_run(num_layers, num_units, lr, reg)
    else:
        assert (False)

    return os.path.join(path_to_results, error_file_name)





def aggregate_RMSE(experiment_name,
                   methods,
                   prediction_horizons=[1, 10, 100, 1000],
                   history_lengths=[1, 10]):
    error_means = pd.DataFrame(columns=["method", "prediction_horizon", "history_length", "setup", "RMSE"])
    for prediction_horizon in prediction_horizons:
        for history_length in history_lengths:
            for method in methods:
                    address = path_to_error_file(method,
                                                 experiment_name,
                                                 prediction_horizon,
                                                 history_length)
                    errors_dict = np.load(address)
                    for setup, errors in errors_dict.items():
                        np_errors = get_angle_errors(errors)
                        mean_error = compute_RMSE_from_errors(np_errors)
                        mean = pd.DataFrame({"method": [method], "prediction_horizon":prediction_horizon, "history_length":[history_length], "setup":[setup], "RMSE": [mean_error]})
                        error_means = error_means.append(mean, ignore_index = True)
            print("Prediction Horizon: {}, History length: {}".format(prediction_horizon, history_length))
    return error_means
