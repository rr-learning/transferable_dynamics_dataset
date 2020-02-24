import numpy
import numpy as np
import sympy as sym
import tensorflow as tf
import sys
import os
import glob
import pickle
from collections import namedtuple, defaultdict, OrderedDict
from sympy.utilities.lambdify import lambdify


from eql.data_utils import get_train_input_fns_from_ndarrays, extract_metadata_from_array, input_from_data
from eql.evaluation import EvaluationHook
from eql.utils import get_run_config, save_results, update_runtime_params, \
                  get_div_thresh_fn,  \
                  tensorboard_summarize, evaluate_learner
from eql.model import ModelFn as EQLModelFn


from DL import DynamicsLearnerInterface
from DL.utils import unrollTrainingData

class EQL(DynamicsLearnerInterface):
    def __init__(self,
                 history_length,
                 prediction_horizon,
                 model_arch_params,
                 model_train_params,
                 optional_params,
                 difference_learning = True):
        super().__init__(history_length, prediction_horizon)
        self.eps = 1e-7
        self.new_data = True
        self._parse_arch_params(**model_arch_params)

        self._parse_train_params(**model_train_params)

        default_params = {
            "output_bound": None,
            "penalty_bounds": None,
            "generate_symbolic_expr": True,
            "id": 1,
            "keys": None,
            "network_init_seed": None,
            "kill_tensorboard_summaries_and_checkpoints": False,
            "use_cluster": True,
            "val_acc_thresh": 0.98,
            "weight_init_param": 1.0
        }
        for key, val in optional_params.items():
            if key in default_params:
                default_params[key] = val
            else:
                raise AttributeError('There are no parameter with name {}'.format(key))
        self.optional_params = default_params
        # If all the params presents we can add them additionaly in self.params
        self.params = {**model_arch_params, **model_train_params, **self.optional_params}
        run_config = get_run_config(self.params["kill_tensorboard_summaries_and_checkpoints"])
        evaluation_hook = EvaluationHook(store_path=self.model_dir)
        self.model_fn = EQLModelFn(config=run_config, evaluation_hook=evaluation_hook)
        self.fast_estimator = FastPredict(tf.estimator.Estimator(model_fn=self.model_fn, config=run_config,
                                                model_dir=self.model_dir,
                                                params=self.params))

    def _parse_arch_params(self, num_h_layers, layer_width):
        self.num_h_layers = num_h_layers
        self.layer_width = layer_width

    def _parse_train_params(self, batch_size, learning_rate, beta1,
                            epochs_first_reg, epochs_per_reg,
                            L0_beta, reg_scales, test_div_threshold, train_val_split,
                            model_dir, evaluate_every):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.batch_size = batch_size
        self.epochs_first_reg = epochs_first_reg
        self.epochs_per_reg = epochs_per_reg
        self.evaluate_every = evaluate_every
        self.L0_beta = L0_beta
        self.reg_scales = reg_scales
        self.test_div_threshold = test_div_threshold
        self.train_val_split = train_val_split
        self.model_dir = model_dir

    def _learn(self, training_inputs, training_targets):
        """
        Parameters
        ----------
        training_inputs:        np-array of shape nTrainingInstances x input dim
                                that represents the input to the dynamics
                                (i.e. relevant observations and actions within
                                the history length and prediction horizon)
        training_targets:       np-array of shape nTrainingInstances x state dim
                                denoting the targets of the dynamics model.
        """
        # states_and_acts = np.concatenate((states, actions), axis=1)
        # inputs, outputs = connected_shuffle([states_and_acts, deltas])
        data = (training_inputs, training_targets)
        metadata = extract_metadata_from_array(train_val_data=data, test_data=None, **self.params)
        self.model_fn.set_metadata(metadata=metadata)
        logging_hook = tf.train.LoggingTensorHook(tensors={'train_accuracy': 'train_accuracy'}, every_n_iter=1000)
        train_input, val_input = get_train_input_fns_from_ndarrays(num_epochs=self.evaluate_every,
        inputs=training_inputs, outputs=training_targets, **self.params, **metadata)
        print('One train episode equals %d epochs.' % self.evaluate_every)
        for i, reg_scale in enumerate(self.reg_scales):
            print('Regularizing with scale %s' % str(reg_scale))
            self.model_fn.set_reg_scale(reg_scale)
            if i == 0:
                max_episode = self.epochs_first_reg // self.evaluate_every
            else:
                max_episode = self.epochs_per_reg // self.evaluate_every
            for train_episode in range(1, max_episode + 1):
                print('Regularized train episode with scale %s: %d out of %d.' % (str(reg_scale), train_episode, max_episode))
                self.fast_estimator.estimator.train(input_fn=train_input, hooks=[logging_hook])
                val_results = self.fast_estimator.estimator.evaluate(input_fn=val_input, name='validation')
                # if (i == 0) and (val_results['eval_accuracy'] > self.params['val_acc_thresh']):
                #     print('Reached accuracy of %d, starting regularization.' % val_results['eval_accuracy'])
                #     break
    def load_normalization_stats(self, observation_sequences, action_sequences):
        targets, inputs = unrollTrainingData(observation_sequences,
                action_sequences, self.history_length, self.prediction_horizon,
                self.difference_learning, self.averaging)
        data = (inputs, targets)
        metadata = extract_metadata_from_array(train_val_data=data, test_data=None, **self.params)
        self.model_fn.set_metadata(metadata=metadata)
        super().load_normalization_stats(observation_sequences, action_sequences)

    def _predict(self, single_input):
        """
        Parameters
        ----------
        single_input:           two dimensional np-array with all the inputs to
                                the dynamics model concatenated (size: input dim)
                                (i.e. relevant observations and actions within
                                the history length and prediction horizon)
        Outputs
        ----------
        observation_prediction: two dimensional np-array  shape: (n_examples, observation_dimension)
                                corresponding the prediction for the observation
                                after 1 step.

        """
        if self.new_data:
            self.fast_estimator.first_run = True
        self.new_data = False
        single_input = single_input.astype(np.float32)
        batch_size = single_input.shape[0]
        if self.model_fn.evaluation_hook.numba_expr is not None:
            predictions = self.model_fn.evaluation_hook.numba_expr(*single_input.T)
            predictions = np.asarray(predictions).T
            return predictions
        else:
            predictions = np.asarray([prediction for prediction in self.fast_estimator.predict(single_input)])
            return predictions

    def name(self):
        return 'EQL' #TODO: change to attribute

    def save(self, model_filename):
        """
        Parameters
        ----------
        filename:   string used as filename to load a model.
        """
        expr = self.model_fn.evaluation_hook.numba_expr

        with open(model_filename, 'wb') as handle:
            pickle.dump(expr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, model_filename):
        """
        Parameters
        ----------
        filename:   string used as filename to save a model.
        """
        with open(model_filename, 'rb') as handle:
            expr = pickle.load(handle)
        self.model_fn.evaluation_hook.numba_expr = expr






class FastPredict:
    """
        Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
        It does this by creating a python generator to keep the predict call open.

        Usage: Just warp your estimator in a FastPredict. i.e.
        classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn, model_dir=model_params.model_dir))

        Author: Marc Stogaitis
     """
    def _createGenerator(self):
        def generator():
            while not self.closed:
                yield (self.next_data, self.next_data)
        self.generator = generator

    def __init__(self, estimator):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self._createGenerator()

    def predict(self, input_fn):
        self.next_data = input_fn
        batch_size = self.next_data.shape[0]
        if self.first_run:
            def input_func():
                ds = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32),
                                                    output_shapes=(tf.TensorShape(self.next_data.shape),
                                                                   tf.TensorShape(self.next_data.shape)))

                value = ds.make_one_shot_iterator().get_next()
                return value

            self.predictions = self.estimator.predict(input_fn=input_func)
            self.first_run = False
        results = [next(self.predictions) for i in range(batch_size)]
        return np.array(results)

    def close(self):
        self.closed = True
        next(self.predictions)
