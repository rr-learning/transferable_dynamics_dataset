import numpy as np
from collections import defaultdict
from DL import DynamicsLearnerInterface
from DL.utils.data_loading import unrollForDifferenceTraining,\
        subsampleTrainingSet, concatenateActionsStates
from pilco.models import PILCO

class PilcoDynamicsLearner(DynamicsLearnerInterface):

    def __init__(self, ninducing_points=None, ntraining_points=None):
        self.ninducing = ninducing_points
        self.ntraining = ntraining_points
    
    def learn(self, obs_seqs, actions_seqs):
        self._check_learning_inputs(obs_seqs, actions_seqs)
        assert self.ntraining
        targets, inputs = unrollForDifferenceTraining(obs_seqs, actions_seqs)
        inputs, targets = subsampleTrainingSet(inputs, targets, self.ntraining)

        # Full GP if no inducing points are provided.
        self.pilco_ = PILCO(inputs, targets, self.ninducing)
        self.pilco_.optimize_models(disp=True)

    def predict(self, obs_hist, action_hist, action_fut):
        self._check_prediction_inputs(obs_hist, action_hist, action_fut)
        assert self.pilco_, "a trained model must be available"
        _, obs_dim  = obs_hist.shape
        _, action_dim = action_hist.shape
        pred_len, _ = action_fut.shape
        predictions = np.zeros((pred_len + 1, obs_dim))
        last_input = concatenateActionsStates(action_hist[-1], obs_hist[-1])
        last_obs = obs_hist[-1]
        for i in range(pred_len):
            predictions[i] = last_obs + self.predict_on_single_input(last_input)
            last_input = concatenateActionsStates(action_fut[i], predictions[i])
            last_obs = predictions[i]
        predictions[i + 1] = last_obs + self.predict_on_single_input(last_input)
        self._check_prediction_outputs(action_fut, predictions)
        return predictions

    def predict_on_single_input(self, action_state_input):
        assert self.pilco_, "a trained model must be available"
        prediction = []
        action_state_input = action_state_input.reshape((1,-1))  # 1xD.
        for model in self.pilco_.mgpr.models:
            mean, _ = model.predict_f(action_state_input)
            prediction.append(mean.item())
        return np.array(prediction)

    def load(self, filename):
        params_dict = np.load(filename)
        for k in params_dict.keys():
            print(k, params_dict[k].shape)
        raise NotImplementedError  # TODO: parse the hyperparameters.

    def save(self, filename):
        """
        Stores the hyperparameters of the GPs which includes inducing points
        in the case of sparse approximations.
        """
        params_dict = defaultdict(list)
        for model in self.pilco_.mgpr.models:
            params = model.read_trainables()
            for key in params.keys():
                params_dict[key].append(params[key])
        np.savez(filename, **params_dict)

