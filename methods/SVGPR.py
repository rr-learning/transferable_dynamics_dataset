"""
Using the implementation of scalable GPs using SVI available in GPflow which is
based on the paper Gaussian Processes for Big Data (Hensman et al. 2013).
"""

import argparse
import gpflow
import numpy as np
from collections import defaultdict
from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData

class SVGPR(DynamicsLearnerInterface):


    class Logger(gpflow.actions.Action):
        def __init__(self, model):
            self.model = model
            self.logf = []

        def run(self, ctx):
            if (ctx.iteration % 100) == 0:
                # Extract likelihood tensor from Tensorflow session
                likelihood = - ctx.session.run(self.model.likelihood_tensor)
                # Append likelihood value to list
                print("Iteration {} loglikelihood {}".format(ctx.iteration,
                        likelihood))
                self.logf.append(likelihood)


    def __init__(self, history_length, prediction_horizon,
            ninducing_points, minibatch_size, niterations, averaging=True,
            streaming=False):
        super().__init__(history_length, prediction_horizon,
                averaging=averaging, streaming=streaming)
        self.ninducing_points = ninducing_points
        self.minibatch_size = minibatch_size
        self.niterations = niterations

    def _learn(self, training_inputs, training_targets):
        ntraining, input_dim  = training_inputs.shape
        kern = gpflow.kernels.RBF(input_dim=input_dim, ARD=True)

        Z = training_inputs[np.random.permutation(
                ntraining)[:self.ninducing_points]].copy()
        assert Z.shape == (self.ninducing_points, input_dim)
        likelihood = gpflow.likelihoods.Gaussian(np.ones(
                self.observation_dimension))

        # Alternatively we can explicitly have one model per dimension.
        self.model_ = gpflow.models.SVGP(training_inputs,
                training_targets, kern, likelihood, Z,
                minibatch_size=self.minibatch_size)
        print('Initial loglikelihood: ', self.model_.compute_log_likelihood())
        self.logger_ = self.run_adam_()
        print('Trained loglikelihood: ', self.model_.compute_log_likelihood())

    def _predict(self, inputs):
        assert self.model_, "a trained model must be available"
        mean, _ = self.model_.predict_f(inputs)
        return mean

    def run_adam_(self):
        """
        Utility function running the Adam Optimiser interleaved with a `Logger` action.

        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimiser action
        adam = gpflow.train.AdamOptimizer().make_optimize_action(self.model_)
        # Create a Logger action
        self.logger = self.Logger(self.model_)
        actions = [adam, self.logger]
        # Create optimisation loop that interleaves Adam with Logger
        loop = gpflow.actions.Loop(actions, stop=self.niterations)()
        # Bind current TF session to model
        self.model_.anchor(self.model_.enquire_session())
        return self.logger

    def name(self):
        return "SVGPR"

    def save(self, filename):
        """
        Stores the trainable hyperparameters of SVGPR including inducing points
        """
        params = self.model_.read_trainables()
        np.savez(filename, **params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--save", help="Filename to save the model")
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)
    dynamics_model = SVGPR(1, 1, niterations = 10, ninducing_points = 10,
            minibatch_size=1000)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())

    if args.save:
        dynamics_model.save(args.save)

    # Plotting the ELBO during optimzation.
    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(-np.array(dynamics_model.logger.logf))
        plt.xlabel('iteration')
        plt.ylabel('ELBO')
        plt.show()

