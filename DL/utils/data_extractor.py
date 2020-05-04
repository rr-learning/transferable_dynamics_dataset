""" Script to extract data splits (i.e., training, testing and validation)
    from a raw npz data file."""

import argparse
import numpy as np
from collections import defaultdict


def split_into_train_val_test(data, nvalidation=3, ntesting=9, ntraining=38):
    training = {}
    validation = {}
    testing = {}
    for key in data.keys():
        testing[key] = data[key][:ntesting, :, :]
    for key in data.keys():
        validation[key] = data[key][ntesting:ntesting + nvalidation, :, :]
    for key in data.keys():
        training[key] = data[key][ntesting + nvalidation:, :, :]
        assert training[key].shape[0] == ntraining
    return training, validation, testing

def from_npz_to_dict(npz):
    ret = {}
    for k in npz.keys():
        if k == "index":
            continue
        ret[k] = npz[k]
    return ret

def discard_prefix(data, discard_prefix):
    """ Removes the prefix of each rollout."""
    for key in data.keys():
        data[key] = data[key][:, discard_prefix:, :]
    return data

def extract_rollouts_from(source_dict, start, end):
    target_dict = {}
    for key in source_dict.keys():
        target_dict[key] = source_dict[key][start:end, :, :]
    return target_dict

def extract(args, full_data):
    """ Extracts dataset splits (training, validation, test, etc.) according
	to the way sines_full.npz and GPs_full.npz were recorded."""
    full_data = discard_prefix(full_data, args.discard_prefix)
    training, validation, testiid = split_into_train_val_test(
            extract_rollouts_from(full_data, 0, 50))
    testtransfer_datasets = []
    for i in range(1, 4):
        _, _, transfertest = split_into_train_val_test(
                extract_rollouts_from(full_data, i * 50, (i + 1) * 50))
        testtransfer_datasets.append(transfertest)
    return training, validation, testiid, testtransfer_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_data", help="filename of the raw data npz",
            default="sines_full.npz", required=True)
    parser.add_argument("--discard_prefix", type=int,
            default=1000, help="discard this many number of observations at"
            "the beginning of each rollout")

    # Output splits
    parser.add_argument("--training", help="Filename training set",
            default="sines_training.npz")
    parser.add_argument("--testiid", help="Filename iid test set",
            default="sines_test_iid.npz")
    parser.add_argument("--validation", help="Filename validation set",
            default="sines_validation.npz")
    parser.add_argument("--testtransfer", help="Filename transfer test sets",
            default="sines_test_transfer_{}.npz")

    args = parser.parse_args()

    raw_data = from_npz_to_dict(np.load(args.raw_data))
    training, validation, testiid, testtransfer_sets = extract(args, raw_data)

    np.savez(args.training, **training)
    np.savez(args.validation, **validation)
    np.savez(args.testiid, **testiid)
    for i, dataset in enumerate(testtransfer_sets):
        np.savez(args.testtransfer.format(i + 1), **dataset)

