"""
Script to tune hyperparameters of model with optuna.

The script so far is constructed to resume a pretrained 
model and only tunes the second level of detail as the 
number of grid points and epsilon only affect the local deformer.
"""

import argparse
import json
import os
import random
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys

from argparse import Namespace

import utils.dataloader as dl
import utils.utility_functions as utils

from deformer.definitions import LOSSES, OPTIMIZERS
from deformer.deformation_layer import RadialBasisFunctionSampler3D
from deformer.latents import LoDPCA
from deformer.metrics import ChamferDistKDTree, SurfaceDistance
from utils.engine import evaluate_meshes, generality, train
from utils.training_utilities import get_deformer, initialize_environment, increase_level, get_latents, save_latents


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to model directory."
    )
    args = parser.parse_args()
    return args


def update_args_(args, params):
    """updates args in-place"""
    dargs = vars(args)
    dargs.update(params)


def main(trial=None):
    args = get_parser()

    directory = args.directory
    with open(directory + 'params.json') as f:
        arg_dict = json.load(f)
        args = Namespace(**arg_dict)

    # Set parameters with range
    if trial is not None:
        params = {'epsilon': trial.suggest_float('epsilon', 0.01, 3),
                  'n_grid_p': trial.suggest_int('n_grid_p', 4, 10)}

        update_args_(args, params)

    # Set arguments
    args.directory = directory
    args.log_dir = args.log_dir + \
        f"/epsilon{args.epsilon}_grid{args.n_grid_p}_{random.randint(0,100000)}_tune"

    os.makedirs(args.log_dir, exist_ok=True)

    args, kwargs, device, logger = initialize_environment(args)

    # Get data
    train_mesh, train_points, val_mesh, val_points, _, _ = dl.build(
        args, kwargs, logger)

    # Get deformer
    deformer = get_deformer(args, train_mesh, train_mesh.sampler.n_samples)

    # Resume deformer to only train second lod
    resume_dict = torch.load(
        args.directory + "checkpoint_latest_deformer_best.pth.tar", map_location=device)
    deformer.load_state_dict(resume_dict["deformer_state_dict"])

    args.lod[1] = int(args.n_grid_p)

    # Set RBF epsilon
    if args.irregular_grid:
        irregular_grid_path = train_mesh.dataset.files[train_mesh.sampler.template_idx]
    else:
        irregular_grid_path = None

    for i in range(len(args.lod)):
        deformer.net[i].add_rbf(RadialBasisFunctionSampler3D(irregular_path=irregular_grid_path,
                                                             n_gridpoints=args.lod[i],
                                                             extent=train_mesh.dataset.extent,
                                                             independent_epsilon=args.independent_epsilon,
                                                             epsilon=args.epsilon))

    # Get and add latents
    lat_params = get_latents(args, train_mesh.sampler.n_samples)
    deformer.net[1].add_lat_params(lat_params[1])

    deformer = nn.DataParallel(deformer)
    deformer.to(device)

    # Unfreeze after level 0 to re-train
    deformer.module.set_at_layer(1)
    for i in range(len(args.lod)):
        deformer.module.set_latent_gradient(i, i >= deformer.module.at_layer)
        deformer.module.set_layer_gradient(i, i >= deformer.module.at_layer)

    deformer.train()

    # Max possible unfreezing is based on trained layers.
    unfrozen = len(args.lod) - deformer.module.at_layer
    epochs = unfrozen * args.increase_layer

    optimizer = OPTIMIZERS[args.optim](
        filter(lambda p: p.requires_grad, deformer.module.parameters()), lr=args.lr)

    logger.info(deformer.module)
    logger.info(deformer)

    # Get metrics and loss
    chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1)
    chamfer_dist.to(device)

    distance_loss = chamfer_dist
    criterion = LOSSES[args.loss_type]

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    with torch.set_grad_enabled(True):
        for epoch in range(1, 1 + epochs):
            # Use sequential updates with freezing of lod
            deformer, scheduler, optimizer, unfrozen = increase_level(
                args, epoch, logger, deformer, unfrozen, optimizer, scheduler)

            # Single train epoch
            loss_eval = train(args,
                              deformer,
                              distance_loss,
                              train_points,
                              epoch,
                              device,
                              logger,
                              optimizer
                              )

            if args.lr_scheduler:
                scheduler.step(loss_eval)

    # Save model
    utils.save_checkpoint(
        {
            "epoch": epoch,
            "deformer_state_dict": deformer.module.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        },
        False,
        epoch,
        os.path.join(args.log_dir, "checkpoint_latest"),
        "deformer",
        logger,
    )

    out_dir = os.path.join(args.log_dir, f"train_meshes")
    os.makedirs(out_dir, exist_ok=True)

    # Get model of latent distribution
    template_idx = train_points.sampler.template_idx
    lat_indices = torch.arange(deformer.module.n_shapes)
    lat_indices = lat_indices[lat_indices != template_idx]
    latents = deformer.module.get_lat_params(lat_indices)
    latent_dist = LoDPCA(latents)

    # Get metrics for generality
    chamfer_distance_metric = SurfaceDistance()
    distance_metrics = {"chamfer": chamfer_distance_metric}
    distance_loss = chamfer_dist
    loss = (distance_loss, criterion)

    # Generality loss for distal femur and liver
    if "classifier" in args.data_root:
        # classifier is trained to optimize training loss
        loss = evaluate_meshes(deformer,
                               distance_metrics,
                               train_mesh,
                               device,
                               LOSSES[args.loss_type],
                               logger,
                               out_dir)
    else:
        loss = generality(deformer, device, (val_mesh, val_points), args, logger, loss,
                          distance_metrics, latent_dist, active_lod=len(args.lod), epoch=300)

    save_latents(deformer, train_points, args)
    logger.info(loss)
    return loss


if __name__ == "__main__":
    args = get_parser()

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))
    study_name = "epsilon_grid_tune"  # Unique identifier of the study.
    storage_name = f"sqlite:///{args.directory}/{study_name}.db"

    # Get optuna study
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')

    study.optimize(main, n_trials=200)
