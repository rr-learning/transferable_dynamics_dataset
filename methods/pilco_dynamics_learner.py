import numpy as np
from DL import DynamicsLearnerInterface
from DL.utils.data_loading import unrollForDifferenceTraining,\
        subsampleTrainingSet
from pilco.models import PILCO

class PilcoDynamicsLearner(DynamicsLearnerInterface):

    def __init__(self, ninducing_points, ntraining_points):
        self.ninducing = ninducing_points
        self.ntraining = ntraining_points
    
    def learn(self, obs_seqs, actions_seqs):
        self._check_learning_inputs(obs_seqs, actions_seqs)
        targets, inputs = unrollForDifferenceTraining(obs_seqs, actions_seqs)
        inputs, targets = subsampleTrainingSet(inputs, targets, self.ntraining)
        self.pilco_ = PILCO(inputs, targets, self.ninducing)
        self.pilco_.optimize_models(disp=True)

    def predict(self, obs_hist, action_hist, action_fut):
        assert self.pilco_, "a trained model must be available"
        raise NotImplementedError

