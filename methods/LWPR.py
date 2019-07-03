"""
Learning a LWPR model.
"""

import argparse
import numpy as np
import time
from collections import deque
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
        init_D = 25
        init_alpha = 175
        self.time_threshold = np.inf
        if settings:
            init_D = settings['init_D']
            init_alpha = settings['init_alpha']
            self.time_threshold = settings.get('time_threshold', np.inf)
        self.model_.init_D = init_D * np.eye(self._get_input_dim())
        self.model_.init_alpha = init_alpha * np.eye(self._get_input_dim())

    def _learn(self, training_inputs, training_targets):

        def gen(inputs, targets):
            for i in range(inputs.shape[0]):
                yield targets[i], inputs[i]

        self._learn_from_stream(gen(training_inputs, training_targets),
                training_inputs.shape[0])

    def _learn_from_stream(self, training_generator, generator_size):
        deck = deque(maxlen=100)
        for count in range(generator_size):
            training_target, training_input = next(training_generator)
            assert training_input.shape[0] == self._get_input_dim()
            assert training_target.shape[0] == self.observation_dimension
            time_before_update = time.perf_counter()
            self.model_.update(training_input, training_target)
            elapsed_time = time.perf_counter() - time_before_update
            deck.append(elapsed_time)
            if count and count % 1000 == 0:
                median_time = sorted(deck)[deck.maxlen // 2]
                print('Update time for iter {} is {}'.format(count,
                        median_time))
                if median_time > self.time_threshold:
                    break

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
    parser.add_argument("--history_length", type=int, default=1)
    parser.add_argument("--prediction_horizon", type=int, default=1)
    parser.add_argument("--averaging", dest='averaging', action='store_true')
    parser.add_argument("--no-averaging", dest='averaging',
            action='store_false')
    parser.set_defaults(averaging=False)
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)

    settings = {"init_alpha": 175, "init_D": 25, "time_threshold" : 0.01}

    # Learning in stremaing mode.
    dynamics_model = lwpr_dyn_model(args.history_length,
            args.prediction_horizon, True, args.averaging, streaming=True,
            settings=settings)
    init_train_time = time.perf_counter()
    dynamics_model.learn(observations, actions)
    end_train_time = time.perf_counter()
    print('Training time {}'.format(end_train_time - init_train_time))

    init_pred_time = time.perf_counter()
    dynamics_model.predict(observations[:, :args.history_length],
            actions[:, :args.history_length],
            actions[:, args.history_length: args.history_length + \
                    args.prediction_horizon - 1])
    end_pred_time = time.perf_counter()
    print('Prediction time {}'.format(end_pred_time - init_pred_time))
