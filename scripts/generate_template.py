"""
Adjusted training script to learn ShapeFlow with hub and spokes 
and return hub aligned shapes for template.


Credits:
Chiyu Max Jiang
https://github.com/maxjiang93/ShapeFlow
"""
import argparse
import json
import os
import glob
import time
import trimesh

import utils.utility_functions as utils
import utils.dataloader as dl

from deformer.definitions import LOSSES, OPTIMIZERS
from shapeflow.layers.chamfer_layer import ChamferDistKDTree
from shapeflow.layers.deformation_layer import NeuralFlowDeformer

import torch
import torch.nn as nn


def train_or_eval(
    args,
    deformer,
    chamfer_dist,
    dataloader,
    epoch,
    device,
    logger,
    optimizer,
    vis_loader,
):
    tot_loss = 0
    count = 0
    criterion = LOSSES["l1"]

    with torch.set_grad_enabled(True):
        toc = time.time()
        for batch_idx, data_tensors in enumerate(dataloader):
            tic = time.time()
            # Send tensors to device.
            data_tensors = [t.to(device) for t in data_tensors]
            (ii,
                jj,
                source_pts,
                target_pts,
                rescale
             ) = data_tensors
            mean_rescale = torch.mean(rescale)

            bs = len(source_pts)
            optimizer.zero_grad()

            # Batch together source and target to create two-way loss training.
            # Cannot call deformer twice (once for each way) because that
            # breaks odeint_ajoint's gradient computation. not sure why.
            source_target_points = torch.cat([source_pts, target_pts], dim=0)
            target_source_points = torch.cat([target_pts, source_pts], dim=0)

            source_target_latents = torch.cat([ii, jj], dim=0)
            target_source_latents = torch.cat([jj, ii], dim=0)
            latent_seq = torch.stack(
                [source_target_latents, target_source_latents], dim=1
            )
            deformed_pts = deformer(
                source_target_points[..., :3], latent_seq
            )  # Already set to via_hub.

            dist = chamfer_dist(
                deformed_pts, target_source_points[..., :3]
            )
            loss = criterion(dist, torch.zeros_like(dist))
            loss.backward()
            # Gradient clipping.
            # torch.nn.utils.clip_grad_value_(
            #    deformer.module.parameters(), True
            # )
            optimizer.step()

            # Check amount of deformation.
            deform_abs = torch.mean(
                torch.norm(deformed_pts - source_target_points, dim=-1)
            )

            tot_loss += loss.item()
            count += bs

            toc = time.time()

            logger.info(
                f"Epoch: {epoch} [{batch_idx * bs}/{len(dataloader) * bs} ({100.0 * batch_idx / len(dataloader):.0f}%)]\t"
                f"Scaled loss: {mean_rescale * loss.item():.6f}\t"
                f"Deform Mean: {mean_rescale * deform_abs.item():.6f}\t"
                f"DataTime: {tic - toc:.4f}\tComputeTime: {time.time() - tic:.4f}\t"
            )
    # generate hub meshes
    with torch.set_grad_enabled(False):
        for ind, data_tensors in enumerate(vis_loader):  # batch size = 1
            ii = torch.tensor([data_tensors[0]], dtype=torch.long)
            jj = torch.tensor([data_tensors[1]], dtype=torch.long)

            source_latents = deformer.module.get_lat_params(ii)
            target_latents = torch.zeros_like(source_latents)
            hub_latents = torch.zeros_like(source_latents)

            data_tensors = [
                t.unsqueeze(0).to(device) for t in data_tensors[2:]
            ]
            vi, fi, vj, fj, rescale = data_tensors
            mean_rescale = torch.mean(rescale)
            vi = vi[0]
            fi = fi[0]
            vj = vj[0]
            fj = fj[0]
            vi_j = deformer(
                vi[..., :3],
                torch.stack(
                    [source_latents, hub_latents, target_latents],
                    dim=1,
                ),
            )
            vi_j *= mean_rescale
            # Save mesh.
            samp_dir = os.path.join(args.log_dir, "deformation_samples")
            os.makedirs(samp_dir, exist_ok=True)
            target_file_name = os.path.basename(
                vis_loader.dataset.files[ii]).split(".")[0]
            trimesh.Trimesh(
                vi_j.detach().cpu().numpy()[0],
                fi.detach().cpu().numpy()[0], process=False
            ).export(os.path.join(samp_dir, f"{target_file_name}_implicit_mean.obj"))


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ShapeNet Deformation Space")
    parser.add_argument(
        "--log_dir", type=str, required=True, help="log directory for run"
    )
    parser.add_argument(
        "--data", type=str, default="femur"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/shapenet_simplified",
        help="path to mesh folder root (default: data/shapenet_simplified)",
    )
    parser.add_argument(
        "--global_extent",
        action="store_true",
        default=False,
        help="Rescale data globally vs. per instance.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Adjust batch size based on the number of gpus available.
    args.batch_size = 14
    use_cuda = torch.cuda.is_available()
    kwargs = (
        {"num_workers": min(12, args.batch_size), "pin_memory": True}
        if use_cuda
        else {}
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    args.global_extent = None

    # Log and create snapshots.
    filenames_to_snapshot = (
        glob.glob("*.py") + glob.glob("*.sh") +
        glob.glob("shapeflow/layers/*.py")
    )
    utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), "w") as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))

    # get data
    train_mesh, train_points = dl.template_build(args, kwargs)

    # Setup model.
    deformer = NeuralFlowDeformer(
        latent_size=128,
        f_width=128,
        s_nlayers=2,
        s_width=5,
        method='dopri5',
        nonlinearity='leakyrelu',
        arch="imnet",
        adjoint=True,
        rtol=1e-4,
        atol=1e-4,
        via_hub=True,
        no_sign_net=True,
        symm_dim=None,
    )

    # Awkward workaround to get gradients from odeint_adjoint to lat_params.
    lat_params = torch.nn.Parameter(
        torch.randn(train_points.dataset.n_shapes, 128) * 1e-1, requires_grad=True
    )
    deformer.add_lat_params(lat_params)
    deformer.to(device)
    all_model_params = list(deformer.parameters())

    optimizer = OPTIMIZERS["adam"](all_model_params, lr=1e-3)

    # More threads don't seem to help.
    chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1)
    chamfer_dist.to(device)
    deformer = nn.DataParallel(deformer)
    deformer.to(device)

    # Training loop.
    for epoch in range(1, 400):
        _ = train_or_eval(
            args=args,
            deformer=deformer,
            chamfer_dist=chamfer_dist,
            dataloader=train_points,
            epoch=epoch,
            device=device,
            logger=logger,
            optimizer=optimizer,
            vis_loader=train_mesh,
        )


if __name__ == "__main__":
    main()
