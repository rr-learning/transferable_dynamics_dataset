"""
Learning a LWPR model.
"""

import argparse
import numpy as np
from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData
from lwpr import LWPR

class lwpr_dyn_model(DynamicsLearnerInterface):

    def __init__(self, history_length, prediction_horizon, difference_learning,
            averaging, streaming, settings=None):
        super().__init__(history_length, prediction_horizon,
                difference_learning, averaging=averaging, streaming=streaming)
        self.model_ = LWPR(self._get_input_dim(), self.observation_dimension)

        # Default values.
        init_D = 20
        init_alpha = 100
        if settings:
            init_D = settings['init_D']
            init_alpha = settings['init_alpha']
        self.model_.init_D = init_D * np.eye(self._get_input_dim())
        self.model_.init_alpha = init_alpha * np.eye(self._get_input_dim())

    def _learn(self, training_inputs, training_targets):

        def gen(inputs, targets):
            for i in range(inputs.shape[0]):
                yield targets[i], inputs[i]

        self._learn_from_stream(gen(training_inputs, training_targets),
                training_inputs.shape[0])

    def _learn_from_stream(self, training_generator, generator_size):
        for count in range(generator_size):
            if count % 1000 == 0:
                print('iter: ', count)
            training_target, training_input = next(training_generator)
            assert training_input.shape[0] == self._get_input_dim()
            assert training_target.shape[0] == self.observation_dimension
            #model_input = training_input.reshape(1, -1)
            self.model_.update(training_input, training_target)

    def _predict(self, inputs):
        assert self.model_, "a trained model must be available"
        prediction = np.zeros((inputs.shape[0], self.observation_dimension))
        for idx in range(inputs.shape[0]):
            prediction[idx, :] = self.model_.predict(inputs[idx])
        return prediction

    def name(self):
        return "LWPR"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)

    # Learning in batch mode.
    dynamics_model = lwpr_dyn_model(1, 1, True, False, False)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())

    # Learning in mini batch mode.
    dynamics_model = lwpr_dyn_model(1, 1, True, False, True)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())
