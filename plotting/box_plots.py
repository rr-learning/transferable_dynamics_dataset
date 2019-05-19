"""
Box plotting of multiple error files.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error_files", required=True, nargs='+',
            help="Filename of the error files to plot")
    parser.add_argument("--names", nargs='+', help="Names of the methods to"
            " display")
    args = parser.parse_args()
    names = args.error_files
    if args.names:
        assert len(args.names) == len(args.error_files)
        names = args.names
    from_setup_to_norms = defaultdict(list)
    for error in args.error_files:
        errors_dict = np.load(error)
        norms_for_setup = []
        for setup, np_errors in errors_dict.items():
            nseq, length, state_dim = np_errors.shape
            joint_errors = np_errors.reshape((-1, state_dim))
            assert np.array_equal(joint_errors[0], np_errors[0,0])
            norms = np.linalg.norm(joint_errors, axis=1)
            from_setup_to_norms[setup].append(norms)
    for setup, norms_array in from_setup_to_norms.items():
        fig, ax = plt.subplots()
        ax.set_title(setup)
        ax.boxplot(norms_array, labels=names)
        plt.show()

