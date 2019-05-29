import os
from time import time
import pickle


import numpy as np


from DL import DynamicsLearnerInterface


atan2, exp, cos, sin, asin, acos, sqrt, tan = np.arctan2, np.exp, np.cos, np.sin, np.arcsin, np.arccos, np.sqrt, np.tan

def logistic(x):
    return 1.0/(1.0+exp(-x))

class Eureqa(DynamicsLearnerInterface):
    def __init__(self,
                 history_length,
                 prediction_horizon):
        super().__init__(history_length, prediction_horizon, averaging = True)
        self.load_normalization_stats()


    def name(self):
        return 'Eureqa'

    def load(self, model_filename):
        pass

    def save(self, model_filename):
        pass

    def learn(self, training_observations, training_actions):
        print("There is no training for Eureqa model")


    def load_normalization_stats(self):
        std_file = "Dataset/dataset_v03_hist{}avg-h{}-standarizers.dat".format(self.history_length, self.prediction_horizon)
        self.inputs_standardizer, self.targets_standardizer = pickle.load(open(std_file,"rb"))

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
        N = single_input.shape[0]
        outputs = np.zeros(shape=(9,N))
        if self.prediction_horizon == 1 and self.history_length == 10:
            (a1,a2,a3,o1,o2,o3,o4,o5,o6,o7,o8, o9) = single_input.T
            outputs[0] = 0.927545926485996*o4 - atan2(o4**2, 0.993772848147297)*atan2(1.51503614021168*o7 - 1.54259547584619*a1, atan2(0.701051028114578*o4**4, 2.18307914600903)) # 38
            outputs[1] = 0.915155783046552*o5 + 0.00999039773492605*a2**3 - 0.00223659206272502 - 0.000142902662532832*o8*a2**4 - 0.00666004702617924*o2*a2**2 # 33
            outputs[2] = 1.85669778663471*o6*logistic(0.504530431798097*a3/o6) - 0.233125311924516*atan2(2.35262483146102*o6*logistic(0.822784787640318*o9*cos(a3)/o6), logistic(-2.21644425141774*o3*o6)) #100
            return outputs.T
        elif self.prediction_horizon == 10 and self.history_length == 1:
            (a1,a2,a3,o1,o2,o3,o4,o5,o6,o7,o8, o9, fa1, fa2, fa3) = single_input.T
            outputs[0] = 0.948119054585915*o4 + 0.386106580966172*fa1 - 0.0685375755464533*o1 - 0.283543691559158*o7 - 0.275025861741297*atan2(atan2(sin(fa1 + 0.744650964153902*o4 - 0.131826320086593*o1 - 0.761602293358418*o7), 0.948119054585915), 0.948119054585915) # 44
            outputs[1] = 0.943758696287776*o5 + 0.251024338492492*a2 + 0.0319433419435836*o9 - 0.0615296919755952*o2 - 0.130351480626544*o8 - 0.152436113810505*sin(0.808347498005865*o5 + 0.227133974920654*fa2)  # 32
            outputs[2] = 0.978015482034113*o6 + 0.187610472853223*a3 + 0.0584523177371514*fa3 - 0.0108895071770413 - 0.0459711720653234*o3 - 0.19277526019455*atan2(o6 + 0.242769987372392*a3 - 0.0122353820290359 - 0.0459711720653234*o3, 0.138588971296687) #36
            return outputs.T
        elif self.prediction_horizon == 10 and self.history_length == 10:
            (a1,a2,a3,o1,o2,o3,o4,o5,o6,o7,o8, o9, fa1, fa2, fa3) = single_input.T
            outputs[0] =  o4 + 0.292515498948442*fa1 - 0.187669935664476*o1 - 0.37346237816614*atan2(0.309877594109472*fa1 + 3.26039998618911*o4**5 - 0.196222096246709*o1, 0.421130317499481) # 36
            outputs[1] =  o5 + 0.184870045531283*fa2 + 0.0935843578392409*a3 - 0.0914628882259398*o2 - 0.208995660626396*atan2(0.220597690264509*fa2 + 0.162184310681982*o9 + 2*o5**3, 0.325811410204256) # 36
            outputs[2] =  0.932820662609382*o6 + 0.421489182928521*a3 - 0.00192261910908924 - 0.368345591341457*atan2(o6, 0.457379554910025) - 0.389121549145925*o9*cos(atan2(o6*o9 + 0.577581276000956*o6*fa3, 0.284430375989894)) # 82
            return outputs.T
        else:
            print("There is no trained model for prediction_horizon={0} and history_length={1}".format(self.history_length, self.prediction_horizon))
