import pandas as pd
import numpy as np
import os
from DL.evaluation.evaluation import get_angle_errors, compute_RMSE_from_errors

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
