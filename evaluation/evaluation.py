"""
Evaluation funcionality. Note that it can also be called as a script.
"""

import argparse
import numpy as np
from DL.utils.data_splitting import CompleteRolloutsDataSplitter
from DL.methods.pilco_dynamics_learner import PilcoDynamicsLearner

class Evaluator(object):

    def __init__(self, data_splitter, model_under_eval, prediction_horizons,
            compute_prediction_error_func, reduce_prediction_error_func,
            min_history=1):
        """
        Parameters
        ----------
        prediction_horizons:            np-array with integer numbers denoting
                                        the different prediction horizons.

        compute_prediction_error_func:  function that computes a particular error
                                        metric between a predicted sequence of
                                        states and the true values.

        reduce_prediction_error_func:   function that computes a single statistic
                                        from a set of prediction errors for a
                                        particular horizon.
        """
        self.data_splitter_ = data_splitter
        self.model_ = model_under_eval
        self.horizons_ = prediction_horizons
        self.min_history_ = min_history
        self.computePredictionError_ = compute_prediction_error_func
        self.reducePredictionErrors_ = reduce_prediction_error_func,

    def computePredictionErrors(self):
        test_obs_seqs, test_act_seqs = self.data_splitter_.get_test_data()
        errors_to_return = []
        for horizon in self.horizons_:
            assert horizon > 0
            horizon_errors = []
            for obs_seq, act_seq in zip(test_obs_seqs, test_act_seqs):
                rollout_len = obs_seq.shape[0]
                assert rollout_len > horizon
                for offset in range(self.min_history_, rollout_len-horizon-1):
                    prediction = self.model_.predict(obs_seq[:offset],
                            act_seq[:offset], act_seq[offset:offset + horizon])
                    error = self.computePredictionError_(prediction,
                            obs_seq[offset:offset + horizon + 1])
                    horizon_errors.append(prediction)
            errors_to_return.append(self.reducePredictionErrors_(current_errors))
        return errors_to_return


def computeErroBasedOnLastPrediction(prediction, ground_truth):
    return np.linalg.norm(prediction[-1] - ground_truth[-1])

def averageErrors(prediction_errors):
    return np.mean(prediction_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="<Required> filename of"
            " the input robot data")
    parser.add_argument("--method", required=True, help="<Required> Name of the"
            " method that will be tested", choices=['pilco'])
    parser.add_argument("--settings", required=True, help="<Required> Settings"
            " for the model (e.g., hyperparameters)")
    args = parser.parse_args()

    # TODO: Find a parameterization for this and load it as a command line arg.
    testing_rollouts = np.arange(3)

    data_splitter = CompleteRolloutsDataSplitter(args.data, testing_rollouts)
    if args.method == 'pilco':

        # TODO: Add the following hyperparameters to the Settings directory
        # and load it as a commmand line argument.
        ninducing = 10
        ntraining = 10
        pilco_dyn = PilcoDynamicsLearner(ninducing, ntraining)
        train_obs_seqs, train_actions_seqs = data_splitter.get_training_data()
        pilco_dyn.learn(train_obs_seqs, train_actions_seqs)

        # Evaluation.
        horizons = np.array([1, 5, 10])
        eval_model = Evaluator(data_splitter, pilco_dyn, horizons,
                computeErroBasedOnLastPrediction, averageErrors)
        errors = eval_model.computePredictionErrors()
        for h, error in zip(horizons, errors):
            print("Error for horizon {} is {}".format(h, error))
    else:
        raise NotImplementedError
