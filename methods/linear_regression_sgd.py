"""
Learning a linear model with SGD using scikit-learn.
"""

import argparse
import numpy as np
from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData
from sklearn import linear_model

class LinearModelSGD(DynamicsLearnerInterface):

    def __init__(self, history_length, prediction_horizon):
        super().__init__(history_length, prediction_horizon)
        self.models_ = []
        for i in range(self.observation_dimension):
            self.models_.append(linear_model.SGDRegressor(
                verbose=1, max_iter=10000000))

    def _learn(self, training_inputs, training_targets):
        for i in range(self.observation_dimension):
            self.models_[i].fit(training_inputs, training_targets[:,i])

    def _predict(self, inputs):
        assert self.models_, "a trained model must be available"
        prediction = []
        for model in self.pilco_.mgpr.models:
            prediction.append(model.predict(inputs))
        return np.hstack(prediction)

    def name(self):
        return "linear-model-SGD"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)
    dynamics_model = LinearModelSGD(1, 1)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())
