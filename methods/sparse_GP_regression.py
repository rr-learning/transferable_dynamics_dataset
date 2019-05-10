""" 
Learning dynamics using sparse Gaussian process regression (SGPR) as in PILCO.
The GP training inputs and training targets are directly taken from the
observed state and action values.
"""

import argparse
import gpflow
import numpy as np
from dynamics_learner_interface import DynamicsLearnerInterface

class SparseGPDynamicsLearner(DynamicsLearnerInterface):

    def __init__(self, state_dims, action_dims, number_inducing_points):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.model = None
        self.number_inducing_points = number_inducing_points

    def get_kernel(self):
        # TODO: ARD.
        return gpflow.kernels.RBF(self.state_dims + self.action_dims)

    def get_inducing_inputs(self, X):
        """Returns a random subset of rows of size m from X."""
        n, d = X.shape
        assert self.number_inducing_points <= n
        Z = X.copy()
        p = np.random.permutation(n)[:self.number_inducing_points]
        return Z[p]

    def get_sparse_model(self, X, Y):
        return gpflow.models.SGPR(X, Y, kern=self.get_kernel(),
                Z=self.get_inducing_inputs(X))

    def get_training_data_from_single_rollout(self, observations, actions):
        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[1] == self.state_dims
        assert actions.shape[1] == self.action_dims

        # The following pairing depends on the way the data was captured.
        # An alternative is to hstack (observations[:-1], actions[:-1]).
        X = np.hstack((observations[:-1], actions[1:]))
        Y = observations[1:]
        return X, Y

    def get_training_data_from_multiple_rollouts(self, observation_sequences,
            action_sequences):
        assert observation_sequences.shape[0] == action_sequences.shape[0]
        nseq = action_sequences.shape[0]

        # Pooling the data across rollouts.
        Xs = []
        Ys = []
        for i in range(nseq):
            X, Y = self.get_training_data_from_single_rollout(
                    observation_sequences[i], action_sequences[i])
            Xs.append(X)
            Ys.append(Y)
        return np.vstack(Xs), np.vstack(Ys)

    def learn(self, observation_sequences, action_sequences):
        X, Y = self.get_training_data_from_multiple_rollouts(
                observation_sequences, action_sequences)
        self.model = self.get_sparse_model(X, Y)
        opt = gpflow.train.ScipyOptimizer()
        print(self.model.compute_log_likelihood())
        opt.minimize(self.model, disp=True, maxiter=100)
        print(self.model.compute_log_likelihood())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
            help="<Required> Path to the input dataset")
    parser.add_argument("--ninducing", type=int, default=1000,
            help="<Required> Path to the input dataset")
    args = parser.parse_args()
    data = np.load(args.dataset)
    observation_sequences = np.concatenate((data['measured_angles'],
            data['measured_velocities'], data['measured_torques']), 2)
    action_sequences = data['constrained_torques']
    SGPDL = SparseGPDynamicsLearner(observation_sequences.shape[-1],
            action_sequences.shape[-1], args.ninducing)
    SGPDL.learn(observation_sequences, action_sequences)
