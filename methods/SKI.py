# -*- coding: utf-8 -*-


import math
import torch
import gpytorch
from matplotlib import pyplot as plt


import argparse
import numpy as np
from dynamics_learner_interface import DynamicsLearnerInterface




class SKIDynamicsLearner(DynamicsLearnerInterface):

    def __init__(self, state_dims, action_dims, learningRate=0.1,
                 trainingIterations=1000):
        self.learningRate=learningRate
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.model = None

    def setAdam(learningRate=0.1):
        self.optimizer = torch.optim.Adam(
                [{'params': self.model.parameters()}],
                lr=learningRate)


    def learn(self, observation_sequences, action_sequences):
        # get data and order them correctly
        X, Y = self.get_training_data_from_multiple_rollouts(
                observation_sequences, action_sequences)        
        # create model
        self.model = GPRegressionModel(X, Y)
        # set model into training mode
        self.model.train()
        self.model.likelihood.train()
        # initialize optimizer and optimization target
        self.setAdam(self.learningRate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood,
                self.model)
        # do training
        for i in range(self.trainingIterations):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, Y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iterations, loss.item()))
            optimizer.step()
        # TODO: maybe include smarter optimizer or optimization criteria
        
    def predict(self, observation_history, action_history, action_future):


# Set model and likelihood into evaluation mode
model.eval()
likelihood.eval()

# Generate nxn grid of test points spaced on a grid of size 1/(n-1) in [0,1]x[0,1]
n = 10
test_x = torch.zeros(int(pow(n, 2)), 2)
for i in range(n):
    for j in range(n):
        test_x[i * n + j][0] = float(i) / (n-1)
        test_x[i * n + j][1] = float(j) / (n-1)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    pred_labels = observed_pred.mean.view(n, n)

# Calc abosolute error
test_y_actual = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
delta_y = torch.abs(pred_labels - test_y_actual).detach().numpy()

# Define a plotting function
def ax_plot(f, ax, y_labels, title):
    im = ax.imshow(y_labels)
    ax.set_title(title)
    f.colorbar(im)

# Plot our predictive means
f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
ax_plot(f, observed_ax, pred_labels, 'Predicted Values (Likelihood)')

# Plot the true values
f, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
ax_plot(f, observed_ax2, test_y_actual, 'Actual Values (Likelihood)')

# Plot the absolute errors
f, observed_ax3 = plt.subplots(1, 1, figsize=(4, 3))
ax_plot(f, observed_ax3, delta_y, 'Absolute Error Surface')







" JUST CREATING DATA"
# train_x : tensor() nObs x nDim torch.Size([1600, 2])
# train_y : tensor() nObs torch.Size([1600])


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        """
        train_x: tensor containing training data. size nObs x nDim
        train_y: tensor containing observations. size nObs
        likelihood: currently just assumed to be Gaussian (observation model)
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]),
            ), grid_size=grid_size, num_dims=train_x.shape[1]
        )
        self.obsLikelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




