"""
Learning a linear model with least squares.
"""

import argparse
import numpy as np
from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData
from sklearn import linear_model

class LinearModel(DynamicsLearnerInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ = linear_model.LinearRegression()

    def _learn(self, training_inputs, training_targets):
        self.model_.fit(training_inputs, training_targets)

    def _predict(self, inputs):
        assert self.model_, "a trained model must be available"
        return self.model_.predict(inputs)

    def name(self):
        return "linear-model-ls"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)
    dynamics_model = LinearModel(1, 1)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())
