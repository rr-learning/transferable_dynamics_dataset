"""
Using an available python implementation of PILCO to learn dynamics.
Only the funcionality related with the forward model learning is being
used. Download a (forked) version of it at https://github.com/DiegoAE/PILCO.
"""


import argparse
import numpy as np
from dynamics_learner_interface import DynamicsLearnerInterface
from PILCO.pilco.models import PILCO


def get_rollouts(data, initial_offset):
    """
    Helper method to get the rollouts in a suitable format. Namely,
    X stores the state-action pairs and Y stores the resulting next states.
    """
    observations = np.concatenate((data['measured_angles'],
        data['measured_velocities'], data['measured_torques']), 2)
    actions = data['constrained_torques']
    nrollouts, length, state_dim = observations.shape
    X = []
    Y = []
    for r in range(nrollouts):
        for t in range(initial_offset, length - 1):
            X.append(np.hstack((observations[r][t], actions[r][t])))
            Y.append(observations[r][t + 1] - observations[r][t])
    return np.stack(X), np.stack(Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
            help="<Required> Path to the input dataset")
    parser.add_argument("--ninducing", type=int,
            help="<Required> Path to the input dataset")
    args = parser.parse_args()
    data = np.load(args.dataset)
    X, Y = get_rollouts(data, 1000)
    pilco = PILCO(X, Y, args.ninducing)
    pilco.optimize_models(disp=True)

