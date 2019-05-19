"""
Dynamics learning using GPs as in PILCO.
"""

import argparse
import numpy as np
from collections import defaultdict
from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData
from pilco.models import PILCO

class PilcoDynamicsLearner(DynamicsLearnerInterface):

    def __init__(self, history_length, prediction_horizon, ninducing_points,
            nsampled_training_points = None):
        super().__init__(history_length, prediction_horizon)
        self.ninducing_points = ninducing_points
        self.nsampled_training_points = nsampled_training_points

    def _learn(self, training_inputs, training_targets):
        assert self.ninducing_points

        # Subsampling the data if required.
        if self.nsampled_training_points:
            training_inputs, training_targets = self._subsample_training_set(
                    training_inputs, training_targets)

        # Full GP if no inducing points are provided.
        self.pilco_ = PILCO(training_inputs, training_targets,
                self.ninducing_points)
        self.pilco_.optimize_models(disp=True)

    def _predict(self, inputs):
        assert self.pilco_, "a trained model must be available"
        prediction = []
        for model in self.pilco_.mgpr.models:
            means, _ = model.predict_f(inputs)
            prediction.append(means)
        return np.hstack(prediction)

    def _subsample_training_set(self, training_inputs, training_targets):
        assert self.nsampled_training_points
        total_size = training_inputs.shape[0]
        permutation = np.random.permutation(
                total_size)[:self.nsampled_training_points]
        return training_inputs[permutation], training_targets[permutation]

    def name(self):
        return "pilco"

    def load(self, filename):
        params_dict = np.load(filename)
        for k in params_dict.keys():
            print(k, params_dict[k].shape)
        raise NotImplementedError  # TODO: parse the hyperparameters.

    def save(self, filename):
        """
        Stores the hyperparameters of the GPs which includes inducing points
        in the case of sparse approximations.
        """
        params_dict = defaultdict(list)
        for model in self.pilco_.mgpr.models:
            params = model.read_trainables()
            for key in params.keys():
                params_dict[key].append(params[key])
        np.savez(filename, **params_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    parser.add_argument("--model_filename", required=True,
            help="<Required> filename where the model will be saved")
    parser.add_argument("--ninducing", default=10, type=int)
    parser.add_argument("--ntraining", default=10, type=int)
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)
    dynamics_model = PilcoDynamicsLearner(1, 1, args.ninducing, args.ntraining)
    dynamics_model.learn(observations, actions)
    dynamics_model.save(args.model_filename)

    # TODO: figure out why after the next line tensorflow throws an error.
    # This apparently only happens when this is file is executed as a script.
    print(dynamics_model.name())
