import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
import os

from itertools import islice
from itertools import chain

from DL import DynamicsLearnerInterface

MODE = 'MAP'

class BNNLearner(DynamicsLearnerInterface):
    def __init__(self, history_length, prediction_horizon,
                 difference_learning = True, learning_rate=0.1,
                 optim_epochs=1, # TODO: redo 400
                 hidden_units=[100, 100],
                 prior_mean=0, prior_std=1,
                 batch_size=512,
                 predDim=3,  # will only predict the first predDim dimensions
                 averaging=None,
                 streaming=None):
        super().__init__(history_length, prediction_horizon,
        difference_learning, averaging=averaging, streaming=streaming)
        print("streaming = {}".format(streaming))
        print("averaging = {}".format(averaging))
#        self.history_length = history_length
#        self.prediction_horizon = prediction_horizon
#        self.observation_dimension = 9
#        self.action_dimension = 3
#        self.difference_learning = difference_learning
        # BNN tuning parameters
        self.learning_rate = np.loadtxt("learning_rate.csv")
        self.optim_epochs = int(np.loadtxt("optim_epochs.csv"))
        layer_width = int(np.loadtxt("width.csv"))
        layer_depth = int(np.loadtxt("depth.csv"))
        self.hidden_units = [layer_width]*layer_depth
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.batch_size = int(np.loadtxt("batch_size.csv"))
        # create models
        self.predDim = predDim
        if averaging:
            self.input_dim = self.observation_dimension + self.action_dimension
            if prediction_horizon > 1:
                self.input_dim += self.action_dimension
        else:
            self.input_dim = self.history_length*(self.observation_dimension + self.action_dimension) \
                           + (self.prediction_horizon-1)*(self.action_dimension)
        self.output_dim = self.predDim
        self.models_ = []
        self.optims_ = []
        for i in range(self.predDim):
            # create model and append to model list
            layers = []
            input_layer = BNNLayer(self.input_dim,
                                   self.hidden_units[0],
                                   activation='relu',
                                   prior_mean=self.prior_mean,
                                   prior_rho=self.prior_std)
            layers.append(input_layer)
            for i in np.arange(len(self.hidden_units)-1):
                layers.append(BNNLayer(self.hidden_units[i],
                                       self.hidden_units[i+1],
                                       activation='relu',
                                       prior_mean=self.prior_mean,
                                       prior_rho=self.prior_std))
                print("more layers")
            output_layer = BNNLayer(self.hidden_units[-1],
                                    1,
                                    activation='none',
                                    prior_mean=self.prior_mean,
                                    prior_rho=self.prior_std)
            layers.append(output_layer)
            self.models_.append(BNN(layers))
            
            optim = torch.optim.Adam(self.models_[-1].parameters(),
                                     lr=self.learning_rate)
            self.optims_.append(optim)

    def name(self):
        return "BNN"        

    def _learn(self, training_inputs, training_targets):
        print("non-streaming training")
        print(training_inputs.shape)
        print(training_targets.shape)
        Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))
#        print(training_inputs.size())
#        print(training_targets[:, 1].reshape((-1, 1)).shape)
        if self.batch_size is None:
            training_inputs = Var(training_inputs)
            training_targets = Var(training_targets)
            for i in range(self.predDim):
                for i_ep in range(self.optim_epochs):
                    kl, lg_lklh = self.models_[i].Forward(
                        training_inputs, training_targets[:, i].reshape((-1, 1)), 1, 'Gaussian')
                    loss = BNN.loss_fn(kl, lg_lklh, 1)
                    self.optims_[i].zero_grad()
                    loss.backward()
                    self.optims_[i].step()
#                    print("{}.{} / {}.{}".format(i, i_ep, self.observation_dimension, self.optim_epochs))
        else:
            dataSize = training_targets[:, 1].size
            stepsPerEpoch = dataSize / self.batch_size
            nSteps = int(np.ceil(self.optim_epochs * stepsPerEpoch))
            for i_ep in range(nSteps):
                # subsample
                currInputs, currTargets = self._subsample_training_set(
                    training_inputs, training_targets)
                currInputs = Var(currInputs)
                currTargets = Var(currTargets)
                for dim in range(self.predDim):
                    kl, lg_lklh = self.models_[dim].Forward(
                        currInputs, currTargets[:, dim].reshape((-1, 1)), 1, 'Gaussian')
                    loss = BNN.loss_fn(kl, lg_lklh, 1)
                    self.optims_[dim].zero_grad()
                    loss.backward()
                    self.optims_[dim].step()
#                    print("{}.{} / {}.{}".format(i_ep, dim, nSteps, self.observation_dimension))
    def _getNewData(self, generator):
       """
       returns chunks of training data and corresponding targets
       """
       currData = np.asarray(list(islice(generator, self.batch_size)))
       newDataTargets = np.fromiter(chain.from_iterable(currData[:, 0]),
                                    dtype=np.float).reshape([512, self.observation_dimension])
       newDataTrain = np.fromiter(chain.from_iterable(currData[:, 1]),
                                  dtype=np.float).reshape([512, self.input_dim])
       return newDataTrain, newDataTargets #

    def _learn_from_stream(self, generator, datastream_size):
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
        print("streaming training")
        print("data stream size: {}".format(datastream_size))
        print("object size 1: {}".format(next(generator)[0].shape))
        print("object size 2: {}".format(next(generator)[1].shape))
        Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))

        dataSize = datastream_size
        stepsPerEpoch = dataSize / self.batch_size
        nSteps = int(np.ceil(self.optim_epochs * stepsPerEpoch))
        print("steps: {}".format(nSteps))
        for i_ep in range(nSteps):
            # subsample
            currInputs, currTargets = self._getNewData(generator)
            currInputs = Var(currInputs)
            currTargets = Var(currTargets)
            for dim in range(self.predDim):
                kl, lg_lklh = self.models_[dim].Forward(
                    currInputs, currTargets[:, dim].reshape((-1, 1)), 1, 'Gaussian')
                loss = BNN.loss_fn(kl, lg_lklh, 1)
                self.optims_[dim].zero_grad()
                loss.backward()
                self.optims_[dim].step()

#        for count in range(self.epochs_ * datastream_size):
#            training_target, training_input = next(generator)
#            assert training_input.shape[0] == self._get_input_dim()
#            model_input = training_input.reshape(1, -1)
#            
#            for output_idx in range(self.observation_dimension):
#                model_target = training_target[output_idx:output_idx + 1]
#                self.models_[output_idx].partial_fit(model_input, model_target)
#        
#        
#        
#
#        input_output_shapes = ([self.input_dim], [self.output_dim])
#        input_output_dtypes = (tf.float64, tf.float64)
#        def switch_input_target():
#            def gen():
#                for target, input in generator:
#                    yield input, target
#            return gen
#        ds = tf.data.Dataset.from_generator(switch_input_target(),
#                input_output_dtypes, input_output_shapes)
#        ds = ds.repeat()
#        ds = ds.batch(self.batch_size)
#        self.model.fit(ds.make_one_shot_iterator(),
#                       steps_per_epoch=datastream_size//self.batch_size,
#                       epochs=self.epochs,
#                       callbacks=[self.tensorboard]
        

    def _subsample_training_set(self, training_inputs, training_targets):
        assert self.batch_size
        total_size = training_inputs.shape[0]
        permutation = np.random.permutation(
                total_size)[:self.batch_size]
        return training_inputs[permutation], training_targets[permutation]         

    def _predict(self, inputs):
#        print("Start prediction")
        prediction = np.zeros((inputs.shape[0], self.observation_dimension))
        Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))
        X_ = Var(inputs)
        for i, model in enumerate(self.models_):
            if MODE == 'MC':
                pred_lst = [model.forward(X_, mode='MC').data.numpy() for _ in range(40)]
                pred = np.array(pred_lst).T            
                prediction[:, i] = pred.mean(axis=2)
            else:
                prediction[:, i] = np.squeeze(model.forward(X_, mode='MAP').data.numpy())
        return prediction

    def load(self, filename):
        """
        Parameters
        ----------
        filename:   string used as filename to load a model.
        """
        raise NotImplementedError

    def save(self, filename):
        """
        Parameters
        ----------
        filename:   string used as filename to save a model.
        """
        if not os.path.exists(filename):
            os.makedirs(filename)
        for i, model in enumerate(self.models_):
            torch.save(model.state_dict(), os.path.join(filename, "state{}.pt".format(i)))

class BNN(nn.Module):
    def __init__(self, layers):
        super(BNN, self).__init__()

        self.layers, self.params = [], nn.ParameterList()
        for layer in layers:
            self.layers.append(layer)
            self.params.extend([*layer.parameters()])   # register module parameters

    def forward(self, x, mode):
        if mode == 'forward':
            net_kl = 0
            for layer in self.layers:
                x, layer_kl = layer.forward(x, mode)
                net_kl += layer_kl
            return x, net_kl
        else:
            for layer in self.layers:
                x = layer.forward(x, mode)
            return x

    def Forward(self, x, y, n_samples, type):

        assert type in {'Gaussian', 'Softmax'}, 'Likelihood type not found'

        # Sample N samples and average
        total_kl, total_likelh = 0., 0.
        for _ in range(n_samples):
            out, kl = self.forward(x, mode='forward')

            # Gaussian output (with unit var)
            # lklh = torch.log(torch.exp(-(y - out) ** 2 / 2e-2) / math.sqrt(2e-2 * math.pi)).sum()

            if type == 'Gaussian':
                lklh = (-.5 * (y - out) ** 2).sum()
            else:   # softmax
                lklh = torch.log(out.gather(1, y)).sum()

            total_kl += kl
            total_likelh += lklh

        return total_kl / n_samples, total_likelh / n_samples

    @staticmethod
    def loss_fn(kl, lklh, n_batch):
        return (kl / n_batch - lklh).mean()

class BNNLayer(nn.Module):
    NegHalfLog2PI = -.5 * math.log(2.0 * math.pi)
    softplus = lambda x: math.log(1 + math.exp(x))

    def __init__(self, n_input, n_output, activation, prior_mean, prior_rho):
        assert activation in {'relu', 'softmax', 'none'}, 'Activation Type Not Found'

        super(BNNLayer, self).__init__()

        # Instantiate a large Gaussian block to sample from, much faster than generating random sample every time
        self._gaussian_block = np.random.randn(10000)

        self.n_input = n_input
        self.n_output = n_output

        self.W_mean = nn.Parameter(torch.ones((n_input, n_output)) * prior_mean)
        self.W_rho = nn.Parameter(torch.ones(n_input, n_output) * prior_rho)

        self.b_mean = nn.Parameter(torch.ones(1, n_output) * prior_mean)
        self.b_rho = nn.Parameter(torch.ones(1, n_output) * prior_rho)

        self.prior_var = Variable(torch.ones(1, 1) * BNNLayer.softplus(prior_rho) ** 2)

        # Set activation function
        self.act = None
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'softmax':
            self.act = F.softmax

        self._Var = lambda x: Variable(torch.from_numpy(x).type(torch.FloatTensor))

    def forward(self, X, mode):
        assert mode in {'forward', 'MAP', 'MC'}, 'BNNLayer Mode Not Found'

        _shape = (X.size()[0], self.n_output)

        # Z: pre-activation. Local reparam. trick is used.
        Z_Mean = torch.mm(X, self.W_mean) + self.b_mean.expand(*_shape)

        if mode == 'MAP': return self.act(Z_Mean) if self.act is not None else Z_Mean

        Z_Std = torch.sqrt(
            torch.mm(torch.pow(X, 2),
                     torch.pow(F.softplus(self.W_rho), 2)) +
            torch.pow(F.softplus(self.b_rho.expand(*_shape)), 2)
        )

        Z_noise = self._random(_shape)
        Z = Z_Mean + Z_Std * Z_noise

        if mode == 'MC': return self.act(Z) if self.act is not None else Z

        # Stddev for the prior
        Prior_Z_Std = torch.sqrt(
            torch.mm(torch.pow(X, 2),
                     self.prior_var.expand(self.n_input, self.n_output)) +
            self.prior_var.expand(*_shape)
        ).detach()

        # KL[posterior(w|D)||prior(w)]
        layer_KL = self.sample_KL(Z,
                                  Z_Mean, Z_Std,
                                  Z_Mean.detach(), Prior_Z_Std)

        out = self.act(Z) if self.act is not None else Z
        return out, layer_KL

    def _random(self, shape):
        Z_noise = np.random.choice(self._gaussian_block, size=shape[0] * shape[1])
        Z_noise = np.expand_dims(Z_noise, axis=1).reshape(*shape)
        return self._Var(Z_noise)

    @staticmethod
    def log_gaussian(x, mean, std):
        return BNNLayer.NegHalfLog2PI - torch.log(std) - .5 * torch.pow(x - mean, 2) / torch.pow(std, 2)

    @staticmethod
    def sample_KL(x, mean1, std1, mean2, std2):
        log_prob1 = BNNLayer.log_gaussian(x, mean1, std1)
        log_prob2 = BNNLayer.log_gaussian(x, mean2, std2)
        return (log_prob1 - log_prob2).sum()
