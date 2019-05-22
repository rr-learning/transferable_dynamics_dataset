""" 
Learning dynamics using sparse Gaussian process regression (SGPR) as in PILCO.
The GP training inputs and training targets are directly taken from the
observed state and action values.
"""

import argparse
import gpflow
import numpy as np
from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData

class SVGPR(DynamicsLearnerInterface):

    def __init__(self, history_length, prediction_horizon,
            ninducing_points, minibatch_size, averaging=True):
        super().__init__(history_length, prediction_horizon,
                averaging=averaging)
        self.ninducing_points = ninducing_points
        self.minibatch_size = minibatch_size

    def _learn(self, training_inputs, training_targets):
        kern = gpflow.kernels.RBF(input_dim=training_inputs.shape[1],
                ARD=True)
        Z = np.random.rand(self.ninducing_points, training_inputs.shape[1])
        self.model_ = gpflow.models.SVGP(training_inputs,
                training_targets, kern, gpflow.likelihoods.Gaussian(),
                Z=Z, minibatch_size=self.minibatch_size)
        # TODO: optimization.
        #pt = gpflow.train.ScipyOptimizer()
        #print(self.model.compute_log_likelihood())
        #opt.minimize(self.model, disp=True, maxiter=200)
        #print(self.model.compute_log_likelihood())

    def _predict(self, inputs):
        assert self.model_, "a trained model must be available"
        mean, _ = self.model_.predict_f(inputs)
        return mean

    def name(self):
        return "SVGPR"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)
    dynamics_model = SVGPR(1, 1, ninducing_points = 10, minibatch_size=10)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())
