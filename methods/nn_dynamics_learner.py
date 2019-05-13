import os
import pickle

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop
import keras.backend as K
from keras.models import load_model

from DL import DynamicsLearnerInterface
from DL.utils.data_loading import DataProcessor
from DL.utils.data_loading import connected_shuffle


class NNDynamicsLearner(DynamicsLearnerInterface):
    def __init__(self, model_arch_params, model_train_params, mode ):


        self.input_dim = 12
        self.output_dim = 9

        self._parse_arch_params(**model_arch_params)
        if mode == "train":
            self._parse_train_params(**model_train_params)
            self.model = Sequential()
            self.build()


    def _parse_arch_params(self, num_layers, num_units, activation):
        self.num_layers = num_layers
        self.size = num_units
        self.activation = activation_from_string(activation)

    def _parse_train_params(self, learning_rate, optimizer, batch_size, validation_split, epochs, loss):
        self.learning_rate = learning_rate
        self.optimizer = optimizer_from_string(optimizer)(lr=self.learning_rate)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.loss = loss

    def build(self):
        all_dims = [self.input_dim] + [self.size] * (self.num_layers - 1)
        for in_dim, size in zip(all_dims[:-1], all_dims[1:]):
            self.model.add(Dense(units=size, input_dim=in_dim, activation=self.activation))
        self.model.add(Dense(units=self.output_dim, input_dim=all_dims[-1], activation=None))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def learn(self, states, actions, delta_states):
        # super()._check_learning_inputs(observation_sequences, action_sequences)
        if states.ndim == 1:
            states = states[None, ...]
        if actions.ndim == 1:
            actions = actions[None, ...]
        states_and_acts = np.concatenate((states, actions), axis=1)
        shuffled_states_and_acts, shuffled_targets = connected_shuffle([states_and_acts, delta_states])

        self.model.fit(x=shuffled_states_and_acts, y=shuffled_targets, batch_size=self.batch_size, epochs=self.epochs,
                       validation_split=self.validation_split, shuffle=True)

    def predict(self, states, actions):
        states_and_acts = np.concatenate((states, actions), axis=1)
        deltas = self.model.predict(states_and_acts)
        return deltas



class NormalizedNNDynamicsLearner(DataProcessor, NNDynamicsLearner):
    def save(self, model_dir):
         if not os.path.exists(model_dir):
            os.makedirs(model_dir)
         self.model.save(os.path.join(model_dir ,"keras_model.h5"))
         normalization_stats = [self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_acts, self.std_acts]
         with open(os.path.join(model_dir ,"normalization_stats.pickle"), 'wb') as handle:
             pickle.dump(normalization_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self, model_dir):
         self.model = load_model(os.path.join(model_dir ,"keras_model.h5"))
         normalization_stats = [self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_acts, self.std_acts]
         with open(os.path.join(model_dir ,"normalization_stats.pickle"), 'rb') as handle:
             normalization_stats = pickle.load(handle)
         self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_acts, self.std_acts = normalization_stats



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
