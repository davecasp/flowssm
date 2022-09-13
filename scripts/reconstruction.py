"""
Run reconstruction experiment
"""

import argparse
import json
import os
import torch
import torch.nn as nn

import utils.dataloader as dl

from argparse import Namespace

from deformer.latents import LoDPCA
from utils.engine import reconstruction
from utils.training_utilities import get_deformer, initialize_environment


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to model directory."
    )
    parser.add_argument(
        "--recon_dir",
        type=str,
        help="Path to reconstruction data."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_parser()

    directory = args.directory
    recon_dir = args.recon_dir

    with open(os.path.join(directory + 'params.json')) as f:
        arg_dict = json.load(f)
        args = Namespace(**arg_dict)

    args.directory = directory
    args.recon_dir = recon_dir
    args.log_dir = os.path.join(args.log_dir, f"reconstruction_meshes")
    os.makedirs(args.log_dir, exist_ok=True)

    # get environment ready
    args, kwargs, device, logger = initialize_environment(args)
    args.at_layer = len(args.lod)

    # get data
    train_mesh, _, _, _, _, _ = dl.build(args, kwargs, logger)
    train_recon, mesh_recon = dl.recon_data(
        args, kwargs, extent=train_mesh.dataset.extent)

    # get deformer
    deformer = get_deformer(args, train_mesh, train_mesh.sampler.n_samples)
    resume_dict = torch.load(
        os.path.join(args.directory + "checkpoint_latest_deformer_best.pth.tar"), map_location=device)
    deformer.load_state_dict(resume_dict["deformer_state_dict"])
    deformer = nn.DataParallel(deformer)
    deformer.to(device)

    # get latent distribution modeled by PCA
    lat_indices = torch.arange(train_mesh.sampler.n_samples)
    template_idx = train_mesh.sampler.template_idx
    lat_indices = lat_indices[lat_indices != template_idx]
    latents = deformer.module.get_lat_params(lat_indices)
    latent_pca = LoDPCA(latents)

    reconstruction(deformer=deformer,
                   device=device,
                   recon_data=(train_recon, mesh_recon),
                   train_data=train_mesh,
                   args=args,
                   logger=logger,
                   latent_distribution=latent_pca,
                   active_lod=len(args.lod),
                   epoch=300,
                   freeze=True)


if __name__ == "__main__":
    main()
