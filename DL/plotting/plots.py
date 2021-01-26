"""
Box plotting of multiple error files.
"""
import argparse
import itertools
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from DL.evaluation.evaluation import get_angle_errors, \
    compute_RMSE_from_errors, get_evaluation_errors


def box_violin_plot(error_files,
                    method_names,
                    violinplot,
                    dataset_names):

    evaluation_errors = {}
    for method_name, error_file in zip(method_names, error_files):
        errors_dict = np.load(error_file)
        evaluation_errors[method_name] = {}
        for dataset_name, errors in errors_dict.items():
            evaluation_errors[method_name][dataset_name] = \
                get_evaluation_errors(errors)

    # normalize and remove delta 0 from plot -----------------------------------
    for dataset_name in evaluation_errors['delta 0'].keys():
        normalizer = np.mean(evaluation_errors['delta 0'][dataset_name])
        for method_name in evaluation_errors.keys():
            evaluation_errors[method_name][dataset_name] /= normalizer
    method_names.remove('delta 0')



    fig, axs = plt.subplots(1, len(dataset_names),
                            sharey=True, figsize=(20,4))

    for i, dataset_name in enumerate(dataset_names):
        ax = axs[i]
        ax.set_title(dataset_name)
        #ax.set_yscale("log")
        # ax.set_ylim(0,0.5)
        evaluation_errors_dataset = \
            [evaluation_errors[method_name][dataset_name]
             for method_name in method_names]
        if violinplot:
            ret = ax.violinplot(evaluation_errors_dataset,
                                showmeans=True, showmedians=True,
                    showextrema=True)
            ret['cmeans'].set_color('r')
            ret['cmedians'].set_color('b')
        else:
            ret = ax.boxplot(evaluation_errors_dataset,
                             showmeans=True)

        ax.set_xticks([y+1 for y in range(len(error_files))])
        ax.set_xticklabels(method_names, rotation=45, fontsize=8)

    return fig

def aggregated_plot(RMSEs,
                    path_to_plots_folder=None,
                    methods = None,
                    prediction_horizons=[1, 10, 100, 1000],
                    history_lengths=[1, 10],
                    setups=["iid_test_data", "transfer_test_data_1", "validation_data"],
                    weighted=True
                    ):
    if methods is not None:
        RMSEs = RMSEs[RMSEs["method"].isin(methods)]

    color_dict = {"training_data": "m",
                  "validation_data": "k",
                  "transfer_test_data_1": "g",
                  "transfer_test_data_2": "b",
                  "transfer_test_data_3": "c",
                  "iid_test_data": "r"}

    label_dict = {"training_data": "train",
                  "validation_data": "validation",
                  "transfer_test_data_1": "transfer 1",
                  "transfer_test_data_2": "transfer 2",
                  "transfer_test_data_3": "transfer 3",
                  "iid_test_data": "iid"}

    marker_dict = {1: ".", 10: "s"}

    fig = plt.figure(figsize=(10*len(prediction_horizons), 5))
    for i, prediction_horizon in enumerate(prediction_horizons):
        ax = fig.add_subplot(1, len(prediction_horizons), i+1)
        filtered = RMSEs[(RMSEs["prediction_horizon"] == prediction_horizon)]
        for history_length in history_lengths:
            for setup in setups:
                filtered_hist_setup = filtered[(filtered["history_length"]==history_length)&(filtered["setup"]==setup)]
                if weighted:
                    delta_0 = float(filtered_hist_setup[filtered_hist_setup["method"]=="delta_0"]["RMSE"])
                    filtered_hist_setup["weighted"] = filtered_hist_setup["RMSE"]/delta_0
                    ax.scatter(filtered_hist_setup["method"], filtered_hist_setup["weighted"],
                               c=color_dict[setup],
                               marker=marker_dict[history_length],
                               label='Hist: {0}, Test: {1}'.format(history_length,
                               label_dict[setup]))
                else:
                    ax.scatter(filtered_hist_setup["method"], filtered_hist_setup["RMSE"],
                               c=color_dict[setup],
                               marker=marker_dict[history_length],
                               label='Hist: {0}, Test: {1}'.format(history_length,
                               label_dict[setup]))
        ax.set_title("Prediction_horizon: {}".format(prediction_horizon))
        plt.legend()
    if path_to_plots_folder is not None:
        if weighted:
            fig.savefig(os.path.join(path_to_plots_folder, "weighted_RMSEs.pdf"))
            fig.savefig(os.path.join(path_to_plots_folder, "weighted_RMSEs.png"))
        else:
            fig.savefig(os.path.join(path_to_plots_folder, "RMSEs.pdf"))
            fig.savefig(os.path.join(path_to_plots_folder, "RMSEs.png"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error_files", required=True, nargs='+',
            help="Filename of the error files to plot")
    parser.add_argument("--names", nargs='+', help="Names of the methods to"
            "display")
    parser.add_argument("--violinplot", action='store_true')
    args = parser.parse_args()
    box_violin_plot(args.error_files, args.names, args.violinplot)


if __name__ == "__main__":
    main()
