"""
Run inference and generate shapes for generality and specificity for validation or test data.
"""

import argparse
import json
import torch
import torch.nn as nn

import utils.dataloader as dl

from argparse import Namespace

from utils.engine import test
from utils.training_utilities import get_deformer, initialize_environment


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to model directory."
    )
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help="Evaluation on test data (default: validation data)."
    )

    args = parser.parse_args()
    return args


def main():
    # get arguments
    args = get_parser()
    directory = args.directory
    test_set = args.test
    with open(directory + 'params.json') as f:
        arg_dict = json.load(f)
        args = Namespace(**arg_dict)
    args.directory = directory
    args.test = test_set

    # Initialize environment
    args, kwargs, device, logger = initialize_environment(args)

    # get data
    train_mesh, _, val_mesh, val_points, test_mesh, test_points = dl.build(
        args, kwargs, logger)

    # get deformer
    args.at_layer = len(args.lod)
    deformer = get_deformer(args, train_mesh, train_mesh.sampler.n_samples)
    resume_dict = torch.load(
        args.directory + "checkpoint_latest_deformer_best.pth.tar", map_location=device)
    deformer.load_state_dict(resume_dict["deformer_state_dict"])
    deformer = nn.DataParallel(deformer)
    deformer.to(device)

    if args.test:
        logger.info("Test data")
        test_data = (test_mesh, test_points)
    else:
        logger.info("Validation data")
        test_data = (val_mesh, val_points)

    # run generality and specificity on evalutation data
    test(deformer=deformer,
         device=device,
         test_data=test_data,
         args=args,
         logger=logger,
         train_data=train_mesh,
         active_lod=len(args.lod),
         epoch=300,
         )


if __name__ == "__main__":
    main()
