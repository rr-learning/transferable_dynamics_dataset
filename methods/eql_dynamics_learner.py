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


class EQL(DynamicsLearnerInterface):
    def __init__(self,
                 history_length,
                 prediction_horizon,
                 model_arch_params,
                 model_train_params,
                 optional_params,
                 difference_learning = True):

        self.eps = 1e-7

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
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, config=run_config,
                                                model_dir=self.model_dir,
                                                params=self.params)
        self.is_trained = False
        super().__init__(history_length, prediction_horizon)

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
        self.results = defaultdict(list)
        logging_hook = tf.train.LoggingTensorHook(tensors={'train_accuracy': 'train_accuracy'}, every_n_iter=1000)
        train_input, val_input = get_train_input_fns_from_ndarrays(num_epochs=self.evaluate_every, inputs=training_inputs, outputs=training_targets, **self.params, **metadata)
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
                self.estimator.train(input_fn=train_input, hooks=[logging_hook])
                val_results = self.estimator.evaluate(input_fn=val_input, name='validation')
                if (i == 0) and (val_results['eval_accuracy'] > self.params['val_acc_thresh']):
                    print('Reached accuracy of %d, starting regularization.' % val_results['eval_accuracy'])
                    break
            self.results = evaluate_learner(learner=self.estimator,
                                            res=self.results,
                                            eval_hook=self.model_fn.evaluation_hook,
                                            val_input=val_input,
                                            test_input=None,
                                            reg_scale=reg_scale)
        self.is_trained = True
        _ = [func for func in dir(self.model_fn) if callable(getattr(self.model_fn, func))]
        self.model_fn.generate_symbolic_expression(3)

    def _predict(self, single_input):
        """
        Parameters
        ----------
        single_input:           one dimensional np-array with all the inputs to
                                the dynamics model concatenated (size: input dim)
                                (i.e. relevant observations and actions within
                                the history length and prediction horizon)
        Outputs
        ----------
        observation_prediction: two dimensional np-array  shape: (1, observation_dimension)
                                corresponding the prediction for the observation
                                after 1 step.

        """

        prediction = self.model_fn.numba_expr(*(single_input.T))
        prediction = np.asarray(prediction).T
        return prediction

    def name(self):
        return 'EQL' #TODO: change to attribute

    def save(self, filename, norm_file=None):
        """
        Parameters
        ----------
        filename:   string used as filename to load a model.
        """
        if norm_file is not None:
            super().save(norm_file)
        exprs = [self.model_fn.sympy_expr, self.model_fn.numba_expr]

        with open(filename, 'wb') as handle:
            pickle.dump(exprs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename, norm_file=None):
        """
        Parameters
        ----------
        filename:   string used as filename to save a model.
        """
        if norm_file is not None:
            super().load(norm_file)
        with open(filename, 'rb') as handle:
            exprs = pickle.load(handle)
        self.model_fn.sympy_expr, self.model_fn.numba_expr = exprs
