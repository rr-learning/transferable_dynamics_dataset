"""
Box plotting of multiple error files.
"""

import argparse
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from DL.evaluation.evaluation import get_angle_errors, compute_RMSE_from_errors


def plot_errors(error_files, names=None, violinplot=False, setups=[]):
    if names:
        assert len(names) == len(error_files)
    from_setup_to_norms = defaultdict(list)
    from_setup_to_RMSEs = defaultdict(list)
    for error in error_files:
        errors_dict = np.load(error)
        for setup, errors in errors_dict.items():
            np_errors = get_angle_errors(errors)
            nseq, length, dim = np_errors.shape
            joint_errors = np_errors.reshape((-1, dim))
            assert np.array_equal(joint_errors[0], np_errors[0,0])
            norms = np.linalg.norm(joint_errors, axis=1)
            from_setup_to_norms[setup].append(norms)
            from_setup_to_RMSEs[setup].append(compute_RMSE_from_errors(
                np_errors))
    plt.figure(figsize=(16,4))
    if len(setups) == 0:
        setups = from_setup_to_norms.keys()
    for i, setup in enumerate(setups):
        norms_array = from_setup_to_norms[setup]
        ax = plt.subplot(1, len(from_setup_to_norms.keys()), i + 1)
        ax.set_title(setup)
        ax.set_yscale("log")
        if violinplot:
            ret = ax.violinplot(norms_array, showmeans=True, showmedians=True,
                    showextrema=True)
        else:
            ret = ax.boxplot(norms_array, showmeans=True)
        if names:
            ax.set_xticks([y+1 for y in range(len(error_files))])
            ax.set_xticklabels(names, rotation=45, fontsize=8)
        red_patch = mpatches.Patch(color='black')
        patches = [red_patch] * len(error_files)
        entries = from_setup_to_RMSEs[setup]
        if names:
            entries = ["{}={:.8f}".format(x,y) for x,y in zip(names, entries)]
        ax.legend(patches, entries, loc='lower left', bbox_to_anchor= (0.0, 1.1))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error_files", required=True, nargs='+',
            help="Filename of the error files to plot")
    parser.add_argument("--names", nargs='+', help="Names of the methods to"
            " display")
    parser.add_argument("--violinplot", action='store_true')
    args = parser.parse_args()
    plot_errors(args.error_files, args.names, args.violinplot)
