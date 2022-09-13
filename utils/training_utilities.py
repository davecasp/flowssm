"""Utility functions to run and train neural flow deformer.
"""

import copy
import json
import numpy as np
import os
import torch
import torch.optim as optim
import utils.dataloader as dl

from torch import nn

from deformer.deformation_layer import NeuralFlowDeformer
from utils.utility_functions import get_logger
from deformer.definitions import OPTIMIZERS


def calc_loss(deformations, targets, distance, criterion):
    """Calculate batch loss from level of details.

    Args:
          deformations (list of tensors): List of LoD deformations [batch, m, 3]
          targets (tensor): [batch, n, 3] points for target
          distance: symmetric distance metric calculated on each batch item
          criterion: Loss criteria (L1 or L2)
          model: deformer model
          weight_decay (float): float weight for latent weight decay
    """
    level_losses = []
    for i in range(len(deformations)):
        dist = distance(deformations[i], targets)
        # average over batch
        level_losses.append(dist.mean(0))
        if i == 0:
            distances = dist
        else:
            distances = dist

    # check if used with surface distance not implemented in torch
    if torch.is_tensor(dist):
        loss = criterion(distances, torch.zeros_like(distances))
    else:
        loss = distances

    return loss, level_losses


def initialize_environment(args):
    """
    Initialize environment for model
    """

    os.makedirs(args.log_dir, exist_ok=True)
    # Adjust batch size based on the number of gpus available.
    if int(torch.cuda.device_count()):
        args.batch_size = args.batch_size
    else:
        args.batch_size = 1
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    kwargs = (
        {"num_workers": min(6, args.batch_size), "pin_memory": True}
        if use_cuda
        else {}
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    args.at_layer = 0

    if args.sample_surface:
        args.samples = 15000
    else:
        args.samples = dl.DATA[args.data]["n_vertices"]

    logger = get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), "w") as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))

    args.n_vis = 10

    # Random seed for reproducability.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data == "liver":
        args.data_root = args.data_root.replace("femur", "liver")

    # calculate the maximum extent for scaling
    args.global_extent = None if args.global_extent else args.global_extent
    return args, kwargs, device, logger


def increase_level(args, epoch, logger, deformer, unfrozen, optimizer, scheduler):
    """ Add level of detail when training.
    """
    if args.increase_layer and unfrozen > 1:
        if (epoch - 1) % args.increase_layer == 0 and not epoch == 1:
            logger.info(
                f"Logging after freezing of layer {((epoch) // args.increase_layer)}.")

            deformer.module.set_latent_gradient(deformer.module.at_layer)
            deformer.module.set_layer_gradient(deformer.module.at_layer)
            deformer.module.set_at_layer(deformer.module.at_layer + 1)

            unfrozen -= 1
            # Re-initialize new optimizer
            optimizer = OPTIMIZERS[args.optim](
                filter(lambda p: p.requires_grad, deformer.module.parameters()), lr=args.lr)
            if args.lr_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min")

            logger.info(deformer.module)
    return deformer, scheduler, optimizer, unfrozen


def get_deformer(args, dataset, n_shapes):
    """Instantiate deformer model.
    """
    if args.rbf:
        # Set number of grid points according to grid type
        if args.irregular_grid:
            irregular_grid_path = dataset.dataset.files[dataset.sampler.template_idx]
        else:
            irregular_grid_path = None

        args.rbf = (irregular_grid_path,
                    dataset.dataset.extent,
                    args.independent_epsilon,
                    args.epsilon,
                    args.lod)
        print(args.rbf)

    # Setup model.
    deformer = NeuralFlowDeformer(
        latent_size=args.lat_dims,
        dim=0 if args.fourier else 3,
        f_width=args.deformer_nf,
        method=args.solver,
        nonlinearity=args.nonlin,
        adjoint=args.adjoint,
        rtol=args.rtol,
        atol=args.atol,
        lod=args.lod,
        at_layer=args.at_layer,
        rbf=args.rbf,
        grid_vector_field=args.grid_vector_field
    )

    # Initialize latent vectors
    if n_shapes is not None:
        lat_params = get_latents(args, n_shapes)
        deformer.add_lat_params(lat_params)

    return deformer


def save_latents(deformer, dataloader, args, name="training_latents"):
    """Save latents with filename.
    """
    latents = {}
    lat_indices = torch.arange(dataloader.sampler.n_samples)
    template_idx = dataloader.sampler.template_idx
    lat_indices = lat_indices[lat_indices != template_idx]
    for idx in lat_indices:
        latents[dataloader.dataset.files[idx]
                ] = deformer.module.get_lat_params(idx)
    torch.save(latents, os.path.join(args.log_dir, f"{name}.pkl"))


def get_latents(args, n_shapes, magnitude=0.1):
    """
    Initialize latent lod vectors.
    """
    if args.grid_vector_field:
        args.lat_dims = 3
        magnitude = 0.0000001

    lat_params = torch.nn.ParameterList([])
    for i in args.lod:
        lat_params.append(torch.nn.Parameter(
            torch.randn(n_shapes, args.lat_dims, i, i, i) * magnitude, requires_grad=True
        ))
    return lat_params


def copy_deformer(deformer, args, train_data, device, n_shapes):
    # Set-up deformer copy (deepcopy fails for non-leave nodes and trained deformer can only handle one set of latents)
    deformer_test = get_deformer(args, train_data, None)
    deformer_test = nn.DataParallel(deformer_test)

    # Add to initialize the state dict keys and correct size of latent parameters.
    model_dict = deformer_test.state_dict()
    pretrained_dict = copy.deepcopy(deformer.state_dict())

    # Filter out all entries that do not exist in the new deformer, i.e. latents
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # Overwrite entries new state dict
    model_dict.update(pretrained_dict)
    deformer_test.load_state_dict(model_dict)

    # Set all network parameter to not require gradients, i.e. freeze network
    for param in deformer_test.module.parameters():
        param.requires_grad = False

    # Initialiaze new embedding
    embedded_latents = get_latents(args, n_shapes, 0.0001)
    embedded_latents.to(device)

    deformer_test.module.add_lat_params(embedded_latents)
    deformer_test.to(device)

    return deformer_test, embedded_latents
