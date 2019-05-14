# -*- coding: utf-8 -*-
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

DTYPE=torch.float

import argparse
import numpy as np
from DL.dynamics_learner_interface.dynamics_learner_interface import DynamicsLearnerInterface
from DL.utils.data_loading import unrollForDifferenceTraining

from DL.utils.standardizer import Standardizer

class SKIDynamicsLearner(DynamicsLearnerInterface):

    def __init__(self, state_dims, action_dims, learningRate=0.1,
                 trainingIterations=100):
        self.learningRate=learningRate
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.models = None
        self.trainingIterations = trainingIterations

    def setAdam(self, currModel, learningRate=0.1):
        self.optimizer = torch.optim.Adam(
                [{'params': currModel.parameters()}],
                lr=learningRate)

    def learn(self, observation_sequences, action_sequences):
        """
        Parameters
        ----------
        observations_sequences: np-array of shape nSequences x nStepsPerRollout x nStates 
                                past state observations
        action_sequences:       np-array of shape nSequences x nStepsPerRollout x nInputs
                                actions taken at the corresponding time points.        
        """
        targets, inputs = unrollForDifferenceTraining(observation_sequences, action_sequences)
        targets = np.asarray(targets, dtype=np.double)
        inputs = np.asarray(inputs, dtype=np.double)
        # standardize everything
        self.targetStandardizer = Standardizer(targets)
        self.inputStandardizer = Standardizer(inputs)
        targets = self.targetStandardizer.standardize(targets)
        inputs = self.inputStandardizer.standardize(inputs)
        targets = torch.from_numpy(targets).float()
        inputs = torch.from_numpy(inputs).float()

        self.models = []
        for modelIndex in np.arange(targets.shape[1]):
            print(modelIndex)
            # create model
            currModel = GPRegressionModel(inputs, targets[:, modelIndex])
            # set model into training mode
            currModel.train()
            currModel.likelihood.train()
            # initialize optimizer and optimization target
            self.setAdam(currModel, self.learningRate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    currModel.likelihood,
                    currModel)
            # do training
            for iterIndex in range(self.trainingIterations):
                self.optimizer.zero_grad()
                output = currModel(inputs)
                loss = -mll(output, targets[:, modelIndex])
                loss.backward()
                print('Iter %d/%d - Loss: %.3f' % (iterIndex + 1, self.trainingIterations, loss.item()))
                self.optimizer.step()
            self.models.append(currModel) 

    def learn2(self, inputs, targets):
        """
        Parameters
        ----------
        observations_sequences: np-array of shape nSequences x nStepsPerRollout x nStates 
                                past state observations
        action_sequences:       np-array of shape nSequences x nStepsPerRollout x nInputs
                                actions taken at the corresponding time points.        
        """
#        targets, inputs = unrollForDifferenceTraining(observation_sequences, action_sequences)
#        targets = np.asarray(targets, dtype=np.double)
#        inputs = np.asarray(inputs, dtype=np.double)
#        # standardize everything
#        self.targetStandardizer = Standardizer(targets)
#        self.inputStandardizer = Standardizer(inputs)
#        targets = self.targetStandardizer.standardize(targets)
#        inputs = self.inputStandardizer.standardize(inputs)
#        targets = torch.tensor(targets, dtype=DTYPE)
#        inputs = torch.tensor(inputs, dtype=DTYPE)

        self.models = []
        for modelIndex in np.arange(targets.shape[1]):
            print(modelIndex)
            # create model
            currModel = GPRegressionModel(inputs, targets[:, modelIndex])
            # set model into training mode
            currModel.train()
            currModel.likelihood.train()
            # initialize optimizer and optimization target
            self.setAdam(currModel, self.learningRate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    currModel.likelihood,
                    currModel)
            # do training
            for iterIndex in range(self.trainingIterations):
                self.optimizer.zero_grad()
                output = currModel(inputs)
                loss = -mll(output, targets[:, modelIndex])
                loss.backward()
                print('Iter %d/%d - Loss: %.3f' % (iterIndex + 1, self.trainingIterations, loss.item()))
                self.optimizer.step()
            self.models.append(currModel)    
      
    def predict(self, observation_history, action_history, action_future):
        """
        Parameters
        ----------
        observation_history:    np-array of shape nStepsPerRollout x nStates
                                all states seen by the system in the current
                                rollout
        action_history:         np-array of shape nStepsPerRollout x nInputs 
                                all actions seen by the system in the current
                                rollout. The last action corresponds to the action
                                that was applied at the final time step.
        action_future:          np-array of shape nPredict x nInputs
                                actions to be applied to the system. First
                                dimension determins prediction horizon. The first
                                action is the action applied one time step after
                                the last action of "action_history".
        Outputs
        ----------
        observation_future:     np-array of shape nPredict+1 x nStates
                                predicted states of the system. The last state
                                will be one time step after the last action of
                                action_future
        """
        # Set model and likelihood into evaluation mode
        self.model.eval()
        self.likelihood.eval()

        x0 = observation_history[-1, :]
        a0 = np.asarray(action_history[-1, :]).reshape([1, -1])
        allActions = np.concatenate([a0, action_future], axis=0)

        observation_future = np.zeros(allActions.shape[0]+1, x0.size)
        observation_future[0, :] = x0

        for i in np.arange(allActions.shape[0]):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(allActions[i, :]))
                currentPrediction = observed_pred.mean # .view(n, n)
            observation_future[i+1, :] = observation_future[i, :] + currentPrediction
        
        return observation_future[1:, :]


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        """
        train_x: tensor containing training data. size nObs x nDim
        train_y: tensor containing observations. size nObs
        likelihood: currently just assumed to be Gaussian (observation model)
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that
#        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
#        print("grid size {}".format(grid_size))
        grid_size = 40 # TODO: MAGIC NUMBER !!!

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




