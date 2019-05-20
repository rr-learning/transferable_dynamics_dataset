import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

from DL import DynamicsLearnerInterface

class BNNLearner(DynamicsLearnerInterface):
    def __init__(self, history_length, prediction_horizon,
                 difference_learning = True, learning_rate=0.1,
                 optim_epochs=400, hidden_units=[100, 100],
                 prior_mean=0, prior_std=1):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.observation_dimension = 9
        self.action_dimension = 3
        self.difference_learning = difference_learning
        # BNN tuning parameters
        self.learning_rate = learning_rate
        self.optim_epochs = optim_epochs
        self.hidden_units = hidden_units
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        # create models
        self.input_dim = self.history_length*(self.observations_dimension + self.action_dimension)
        self.output_dim = self.observation_dim
        self.models_ = []
        self.optims = []
        for i in range(self.observation_dimension):
            # create model and append to model list
            layers = []
            input_layer = BNNLayer(self.input_dimension,
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
                                    self.output_dim,
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
        for i in range(self.observation_dimension):
            for i_ep in range(self.optim_epochs):
                kl, lg_lklh = self.models_[i].Forward(
                    training_inputs, training_targets[:, i], 1, 'Gaussian')
                loss = BNN.loss_fn(kl, lg_lklh, 1)
                self.optims_[i].zero_grad()
                loss.backward()
                self.optims_[i].step()

    def _predict(self, inputs):
        prediction = np.zeros((inputs.shape[0], self.observation_dimension))
        Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))
        X_ = Var(inputs)
        for i, model in enumerate(self.models_):
            pred_lst = [model.forward(X_, mode='MC').data.numpy() for _ in range(500)]
            pred = np.array(pred_lst).T            
            prediction[:, i] = pred.mean(axis=2)
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
        raise NotImplementedError


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
