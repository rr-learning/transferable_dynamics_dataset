"""
Learning a linear model with SGD using scikit-learn.
"""

import argparse
import numpy as np
from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData
from sklearn import linear_model

class LinearModelSGD(DynamicsLearnerInterface):

    def __init__(self, history_length, prediction_horizon, difference_learning,
            averaging, streaming, epochs=1, settings=None):
        super().__init__(history_length, prediction_horizon,
                difference_learning, averaging=averaging, streaming=streaming)
        eta0 = 0.00001
        if settings:
            eta0 = settings['eta0']
        self.models_ = []
        for i in range(self.observation_dimension):
            self.models_.append(linear_model.SGDRegressor(verbose=False,
                learning_rate='constant', eta0=eta0))
        self.epochs_ = epochs

    def _learn(self, training_inputs, training_targets):
        for i in range(self.observation_dimension):
            self.models_[i].fit(training_inputs, training_targets[:,i])

    def _learn_from_stream(self, training_generator, generator_size):
        for count in range(self.epochs_ * generator_size):
            training_target, training_input = next(training_generator)
            assert training_input.shape[0] == self._get_input_dim()
            model_input = training_input.reshape(1, -1)
            for output_idx in range(self.observation_dimension):
                model_target = training_target[output_idx:output_idx + 1]
                self.models_[output_idx].partial_fit(model_input, model_target)

    def _predict(self, inputs):
        assert self.models_, "a trained model must be available"
        prediction = np.zeros((inputs.shape[0], self.observation_dimension))
        for i, model in enumerate(self.models_):
            prediction[:, i] = model.predict(inputs)
        return prediction

    def name(self):
        return "linear-model-SGD"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)

    # Learning in batch mode.
    dynamics_model = LinearModelSGD(1, 1, True, False, False)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())

    # Learning in mini batch mode.
    dynamics_model = LinearModelSGD(1, 1, True, False, True)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())
