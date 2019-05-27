"""
Using the implementation of scalable GPs using SVI available in GPflow which is
based on the paper Gaussian Processes for Big Data (Hensman et al. 2013).
"""

import argparse
import gpflow
import numpy as np
from DL import DynamicsLearnerInterface
from DL.utils import loadRobotData

class SVGPR(DynamicsLearnerInterface):


    class Logger(gpflow.actions.Action):
        def __init__(self, model):
            self.model = model
            self.logf = []

        def run(self, ctx):
            if (ctx.iteration % 10) == 0:
                # Extract likelihood tensor from Tensorflow session
                likelihood = - ctx.session.run(self.model.likelihood_tensor)
                # Append likelihood value to list
                self.logf.append(likelihood)


    def __init__(self, history_length, prediction_horizon,
            ninducing_points, minibatch_size, averaging=True):
        super().__init__(history_length, prediction_horizon,
                averaging=averaging)
        self.ninducing_points = ninducing_points
        self.minibatch_size = minibatch_size

    def _learn(self, training_inputs, training_targets):
        kern = gpflow.kernels.RBF(input_dim=training_inputs.shape[1],
                ARD=True)
        # TODO: Make Z a subset of the training data.
        Z = np.random.rand(self.ninducing_points, training_inputs.shape[1])
        self.model_ = gpflow.models.SVGP(training_inputs,
                training_targets, kern, gpflow.likelihoods.Gaussian(), Z,
                minibatch_size=self.minibatch_size)
        print('Initial loglikelihood: ', self.model_.compute_log_likelihood())
        self.logger_ = self.run_adam_(self.model_, iterations=10000)
        print('Trained loglikelihood: ', self.model_.compute_log_likelihood())

    def _predict(self, inputs):
        assert self.model_, "a trained model must be available"
        mean, _ = self.model_.predict_f(inputs)
        return mean

    def run_adam_(self, model, iterations):
        """
        Utility function running the Adam Optimiser interleaved with a `Logger` action.

        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimiser action
        adam = gpflow.train.AdamOptimizer().make_optimize_action(model)
        # Create a Logger action
        self.logger = self.Logger(model)
        actions = [adam, self.logger]
        # Create optimisation loop that interleaves Adam with Logger
        loop = gpflow.actions.Loop(actions, stop=iterations)()
        # Bind current TF session to model
        model.anchor(model.enquire_session())
        return self.logger

    def name(self):
        return "SVGPR"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_filename", required=True,
            help="<Required> filename of the input robot data")
    args = parser.parse_args()
    observations, actions = loadRobotData(args.data_filename)
    dynamics_model = SVGPR(1, 1, ninducing_points = 10, minibatch_size=1000)
    dynamics_model.learn(observations, actions)
    print(dynamics_model.name())

    # Plotting the ELBO during optimzation.
    import matplotlib.pyplot as plt
    plt.plot(-np.array(dynamics_model.logger.logf))
    plt.xlabel('iteration')
    plt.ylabel('ELBO')
    plt.show()

