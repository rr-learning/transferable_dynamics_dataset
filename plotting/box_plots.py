"""
Box plotting of multiple error files.
"""

import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from DL.evaluation.evaluation import get_angle_errors, compute_RMSE_from_errors

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
    from_setup_to_RMSEs = defaultdict(list)
    for error in args.error_files:
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
    for setup in from_setup_to_norms.keys():
        norms_array = from_setup_to_norms[setup]
        fig, ax = plt.subplots()
        ax.set_title(setup)
        ret = ax.boxplot(norms_array, labels=names)
        ax.legend(ret["boxes"], from_setup_to_RMSEs[setup], loc='upper left')
        #colors = itertools.cycle(['red','green','blue', 'purple', "yellow"])
        #for box in ret["boxes"]:
        #    box.set_facecolor(next(colors))
        plt.show()

