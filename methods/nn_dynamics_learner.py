import os
from time import time

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop
import keras.backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras import regularizers

from DL import DynamicsLearnerInterface




class NNDynamicsLearner(DynamicsLearnerInterface):
    def __init__(self,
                 history_length,
                 prediction_horizon,
                 model_arch_params,
                 model_train_params,
                 mode):


        self.input_dim = 3 * (prediction_horizon - 1) + history_length * 12
        self.output_dim = 9
        super().__init__(history_length, prediction_horizon)
        self._parse_arch_params(**model_arch_params)
        if mode == "train":
            self._parse_train_params(**model_train_params)
            self.model = Sequential()
            self.build()

    def name(self):
        return 'NN' #TODO: change to attribute

    def _parse_arch_params(self, num_layers, num_units, activation):
        self.num_layers = num_layers
        self.size = num_units
        self.activation = activation_from_string(activation)

    def _parse_train_params(self, learning_rate, optimizer, batch_size, validation_split, epochs, loss, l2_reg): #  l2_reg,
        self.learning_rate = learning_rate
        self.optimizer = optimizer_from_string(optimizer)(lr=self.learning_rate)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.loss = loss
        self.l2_reg = l2_reg

    def build(self):
        all_dims = [self.input_dim] + [self.size] * (self.num_layers - 1)
        for in_dim, size in zip(all_dims[:-1], all_dims[1:]):
            self.model.add(Dense(units=size, input_dim=in_dim, activation=self.activation, kernel_regularizer=regularizers.l2(self.l2_reg))) #
        self.model.add(Dense(units=self.output_dim, input_dim=all_dims[-1], activation=None))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.tensorboard = TensorBoard(log_dir="logs/history_length_{4}_prediction_horizon_{5}_n_{0}_m_{1}_l2_reg_{3}_Adam_lr_{2}_bs_512_epochs_400_{4}".format(self.num_layers,
                                                                                 self.size,
                                                                                 self.learning_rate,
                                                                                 self.l2_reg,
                                                                                 self.history_length,
                                                                                 self.prediction_horizon,
                                                                                 time()))

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
        self.model.fit(x=training_inputs, y=training_targets, batch_size=self.batch_size, epochs=self.epochs,
                       validation_split=self.validation_split, shuffle=False, callbacks=[self.tensorboard])

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
        observation_prediction: two dimensional np-array  shape: (n_examples, observation_dimension)
                                corresponding the prediction for the observation
                                after 1 step.

        """


        deltas = self.model.predict(single_input)
        return deltas

    def save(self, model_filename):
        self.model.save(model_filename)


    def load(self, model_filename):
        self.model = load_model(model_filename)


def optimizer_from_string(opt_str):
    opt_dict = {'Adam': Adam, 'Adagrad': Adagrad, 'Adadelta': Adadelta, 'SGD': SGD, 'RMSprop': RMSprop}
    if opt_str in opt_dict:
        return opt_dict[opt_str]
    else:
        raise NotImplementedError('Implement optimizer {} and add it to dictionary'.format(opt_str))


def activation_from_string(act_str):
    act_dict = {'relu': K.relu, 'tanh': K.tanh}
    if act_str in act_dict:
        return act_dict[act_str]
    else:
        raise NotImplementedError('Add activation function {} to dictionary'.format(act_str))
