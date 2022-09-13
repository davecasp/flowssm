"""
Helper functions to run, train and evalutate model and experiments.
"""

import multiprocessing as mp
import numpy as np
import os
import re
import time
import torch
import trimesh

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

from deformer.definitions import LOSSES, OPTIMIZERS
from deformer.latents import LoDPCA
from deformer.metrics import SurfaceDistance, ChamferDistKDTree
from utils.training_utilities import copy_deformer, increase_level, save_latents, calc_loss


def specificity(deformer,
                device,
                train_data,
                args,
                latent_distribution,
                n_samples=1000):
    """
    Sample meshes for specificity by sampling latents from the training distribution.

    Args:
        deformer (torch.nn.model): Neural Flow Deformer.
        device (str): torch device.
        train_data (Dataloader): Dataloader for the training data.
        args (dict): dict of argparse arguments.
        latent_distribution (torch.nn.model): model of the latent distribution.
        n_samples (int): number of shapes to sample.
    """
    _train_loader = train_data
    template_idx = train_data.sampler.template_idx
    train_data = _train_loader.dataset

    out_dir = os.path.join(args.log_dir, f"sampled_meshes")
    os.makedirs(out_dir, exist_ok=True)

    # Sample the latent representations from training distribution
    sampled_latents = latent_distribution.generate_random_samples(
        n_samples, len(latent_distribution))
    deformer.eval()
    with torch.set_grad_enabled(False):
        # Get template shape
        data_tensors = train_data.get_single(template_idx)
        template_shape, source_face, target_scale = data_tensors
        template_shape = template_shape.to(device).unsqueeze(0)
        template_shape = template_shape.float()

        # Generate shapes from samples latents
        for i, target_latents in enumerate(tqdm(sampled_latents)):
            # Apply deformation to template shape
            deformed_pts = deformer(template_shape, target_latents)

            # Save result
            out_file = os.path.join(
                out_dir, f"sampled{i}_{type(latent_distribution).__name__}.ply")
            trimesh.Trimesh(
                (deformed_pts[-1][0] *
                    target_scale).detach().cpu().numpy(),
                source_face.detach().cpu().numpy(),
                process=False).export(out_file)


def generality(deformer,
               device,
               test_data,
               args,
               logger,
               loss,
               distance_metric,
               latent_distribution,
               active_lod=3,
               epoch=60,
               freeze=True):
    """
    Show generality by embedding unseen shapes.

    Args:
        deformer (torch.nn.model): Neural Flow Deformer.
        device (str): torch device.
        test_data (Dataloader): Dataloader for the test data.
        args (dict): dict of argparse arguments.
        logger: logger instance.
        loss (tuple): distance loss and criterion.
        distance_metric: surface distance metric to evaluate generality.
        latent_distribution (torch.nn.model): model of the latent distribution.
        active_lod (int): number of trained level of details.
        epochs (int): number of epochs level of detail.
        freeze(bool): if false then all level of details are trained in parallel,
            else all levels are trained consecutive with intermediate freezing.
    """
    (distance_loss, criterion) = loss

    test_mesh, test_point = test_data
    nshapes = test_mesh.sampler.n_samples
    template_idx = test_mesh.sampler.template_idx

    # Copy deformer to add new embedding
    deformer_test, _ = copy_deformer(
        deformer, args, test_mesh, device, nshapes)

    if not freeze:
        deformer_test.module.set_at_layer(len(args.lod))
    else:
        deformer_test.module.set_at_layer(0)

    deformer_test.train()

    # max possible unfreezing is based on trained layers.
    unfrozen = active_lod
    epochs = active_lod * epoch
    lr = .01  # fixed lr for generality

    optim = OPTIMIZERS[args.optim](
        filter(lambda p: p.requires_grad, deformer_test.module.parameters()), lr=lr)

    logger.info(deformer_test.module)

    with torch.set_grad_enabled(True):
        for epoch in range(1, 1 + epochs):
            # Use sequential freezing to train the levels of detail individually
            deformer_test, _, optim, unfrozen = increase_level(args,
                                                               epoch,
                                                               logger,
                                                               deformer_test,
                                                               unfrozen,
                                                               optim,
                                                               scheduler=None
                                                               )

            for data_tensors in test_point:
                tic = time.time()
                # Send tensors to device.
                data_tensors = [t.to(device) for t in data_tensors]
                (_,
                    jj,
                    source_pts,
                    target_pts,
                    rescale,
                 ) = data_tensors
                mean_rescale = torch.mean(rescale)
                optim.zero_grad()

                # Project data into training space
                target_latents = deformer_test.module.get_lat_params(jj)
                target_latents = latent_distribution.project_data(
                    target_latents)

                # Apply deformation and calculate loss
                deformed_pts = deformer_test(source_pts, target_latents)
                loss, level_losses = calc_loss(
                    deformed_pts, target_pts, distance_loss, criterion)
                loss.backward()
                optim.step()

                logging = ""
                for a, level_loss in enumerate(level_losses):
                    logging += f"Level {a}:{mean_rescale*level_loss} \t"

                # Check amount of deformation.
                deform_abs = torch.mean(torch.norm(
                    deformed_pts[-1] - source_pts, dim=-1))

                toc = time.time()
                dist = level_losses[-1].item()

                logger.info(
                    f"Iter: {epoch}, Scaled loss: {loss.item():.4f}\t"
                    f"Dist: {mean_rescale*dist:.4f}\t"
                    f"Deformation Magnitude: {mean_rescale*deform_abs.item():.4f}\t"
                    f"{logging}"
                    f"Time per iter (s): {toc-tic:.4f}\t"
                )

    # Save embedded test latents
    save_latents(deformer_test, test_point, args, "test_latents")

    lat_indices = torch.arange(nshapes)
    lat_indices = lat_indices[lat_indices != template_idx]
    latents = deformer_test.module.get_lat_params(lat_indices)
    for latent in latents:
        logger.info(
            f"Generality latent distribution –– Mean: {latent.flatten().mean()} (+/-{latent.flatten().std()})")

    # Evaluate surface distance on meshes
    out_dir = os.path.join(args.log_dir, f"generality_meshes")
    os.makedirs(out_dir, exist_ok=True)
    cham_dist = evaluate_meshes(deformer_test,
                                distance_metric,
                                test_mesh,
                                device,
                                criterion,
                                logger,
                                out_dir,
                                latent_distribution=latent_distribution)

    return np.array(cham_dist).mean()


def reconstruction(deformer,
                   device,
                   recon_data,
                   train_data,
                   args,
                   logger,
                   latent_distribution,
                   active_lod=3,
                   epoch=60,
                   freeze=True):
    """
    Find embedding for reconstruction task. This function mainly follows the 
    set-up of the generality experiment.

    Args:
        deformer (torch.nn.model): Neural Flow Deformer.
        device (str): torch device.
        recon_data (Dataloader): Dataloader for the reconstruction data.
        train_data (Dataloader): Dataloader for the training data.
        args (dict): dict of argparse arguments.
        logger: logger instance.
        latent_distribution (torch.nn.model): model of the latent distribution.
        active_lod (int): number of trained level of details.
        epochs (int): number of epochs level of detail.
        freeze(bool): if false then all level of details are trained in parallel,
            else all levels are trained consecutive with intermediate freezing.
    """

    # Initialize Chamfer distance
    chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=16)
    chamfer_dist.to(device)
    distance_loss = chamfer_dist

    recon_point, _ = recon_data
    n_shapes = len(recon_point.dataset)

    deformer_test, _ = copy_deformer(
        deformer, args, train_data, device, n_shapes)

    if not freeze:
        deformer_test.module.set_at_layer(len(args.lod))
    else:
        deformer_test.module.set_at_layer(0)

    # Optimize for latents.
    deformer_test.train()
    toc = time.time()

    # max possible unfreezing is based on trained layers.
    unfrozen = active_lod
    epochs = active_lod * epoch
    lr = .001

    optim = OPTIMIZERS[args.optim](
        filter(lambda p: p.requires_grad, deformer_test.module.parameters()), lr=lr)

    logger.info(deformer_test.module)
    logger.info("Reconstruction task")
    out_dir = args.log_dir

    os.makedirs(out_dir, exist_ok=True)

    reconstructed_meshes = {}
    target_meshes = {}

    with torch.set_grad_enabled(True):
        for epoch in range(1, 1 + epochs):
            # Use sequential freezing of layers of detail
            deformer_test, _, optim, unfrozen = increase_level(args,
                                                               epoch,
                                                               logger,
                                                               deformer_test,
                                                               unfrozen,
                                                               optim,
                                                               scheduler=None)

            for data_tensors in recon_point:
                tic = time.time()
                # Send tensors to device.
                data_tensors = [t.to(device) for t in data_tensors]

                (idx,
                 mean_verts,
                 mean_faces,
                 v_target,
                 f_target,
                 recon_verts,
                 recon_faces,
                 target_scale) = data_tensors
                mean_rescale = torch.mean(target_scale)

                optim.zero_grad()

                # Project data into training space
                target_latents = deformer_test.module.get_lat_params(idx)
                target_latents = latent_distribution.project_data(
                    target_latents)

                deformed_pts = deformer_test(mean_verts, target_latents)

                # Differentiable sample meshes
                deformed_pts_sampled = []
                for i in range(len(deformed_pts)):
                    deformed_meshes = Meshes(
                        verts=deformed_pts[i], faces=mean_faces)
                    deformed_pts_sampled.append(sample_points_from_meshes(
                        deformed_meshes, num_samples=15000))

                _, loss1, loss2 = distance_loss(
                    deformed_pts_sampled[-1], recon_verts, True)

                # Use onesided Chamfer
                chamfer = 1 * loss1.mean() + 0. * loss2.mean()

                loss = chamfer
                loss.backward()

                # Check amount of deformation.
                deform_abs = torch.mean(torch.norm(
                    deformed_pts[-1] - mean_verts, dim=-1))

                optim.step()

                toc = time.time()
                dist = chamfer.item()
                logger.info(recon_point.dataset.source_file[idx])

                logger.info(
                    f"Iter: {epoch}, Scaled loss: {loss.item():.4f}\t"
                    f"Dist: {mean_rescale*dist:.4f}\t"
                    f"Deformation Magnitude: {mean_rescale*deform_abs.item():.4f}\t"
                    f"Time per iter (s): {toc-tic:.4f}\t"
                )
    # Generate reconstructions
    with torch.no_grad():
        for data_tensors in recon_point:
            # Send tensors to device.
            data_tensors = [t.to(device) for t in data_tensors]

            (idx,
             mean_verts,
             mean_faces,
             v_target,
             f_target,
             recon_verts,
             recon_faces,
             target_scale) = data_tensors
            mean_rescale = torch.mean(target_scale)

            optim.zero_grad()

            # Project data into training space
            target_latents = deformer_test.module.get_lat_params(idx)
            target_latents = latent_distribution.project_data(target_latents)

            deformed_pts = deformer_test(mean_verts, target_latents)
            mean_rescale = mean_rescale.detach().cpu().numpy()
            trg_mesh = trimesh.Trimesh(
                v_target.detach().cpu().numpy()[0] * mean_rescale,
                f_target.detach().cpu().numpy()[0],
                process=False
            )

            if (recon_faces.detach().cpu().numpy()[0] == 0).all():
                recon_mesh = trimesh.points.PointCloud(
                    recon_verts.detach().cpu().numpy()[0] * mean_rescale
                )
            else:
                recon_mesh = trimesh.Trimesh(
                    recon_verts.detach().cpu().numpy()[0] * mean_rescale,
                    recon_faces.detach().cpu().numpy()[0],
                    process=False
                )

            deformed_mesh = trimesh.Trimesh(
                deformed_pts[-1].detach().cpu().numpy()[0] * mean_rescale,
                mean_faces.detach().cpu().numpy()[0],
                process=False
            )
            deformed_mesh0 = trimesh.Trimesh(
                deformed_pts[0].detach().cpu().numpy()[0] * mean_rescale,
                mean_faces.detach().cpu().numpy()[0],
                process=False
            )

            # Save meshes with id
            reconstructed_meshes[idx.item()] = deformed_mesh
            target_meshes[idx.item()] = trg_mesh

            recon_mesh.export(os.path.join(
                out_dir, f"{idx.item()}_recon.obj"))
            trg_mesh.export(os.path.join(
                out_dir, f"{idx.item()}.obj"))
            deformed_mesh.export(os.path.join(
                out_dir, f"{idx.item()}_deformed.obj"))
            deformed_mesh0.export(os.path.join(
                out_dir, f"{idx.item()}_deformed_lod0.obj"))

    # Evaluate reconstruction
    sparse = [file for file in recon_point.dataset.source_file if "_sparse" in file]
    tasks = set([re.findall(r"_[0-9]+_sparse", file)[0] for file in sparse])
    tasks.add("partial")
    meshes = {}
    chamfer_distance_metric = SurfaceDistance()
    for task in list(tasks):
        meshes[task] = ([], [])

        for id in reconstructed_meshes.keys():
            file_name = os.path.basename(recon_point.dataset.source_file[id])
            if task in file_name:
                recon = reconstructed_meshes[id]
                tar = target_meshes[id]
                meshes[task][0].append(recon)
                meshes[task][1].append(tar)

        data = zip(meshes[task][0], meshes[task][1])
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(chamfer_distance_metric, data)

        distances = np.concatenate(results)
        logger.info(task)
        logger.info(distances)
        logger.info(distances.mean())
        logger.info(distances.std())


def evaluate_meshes(deformer,
                    distance_metric,
                    test_mesh,
                    device,
                    criterion,
                    logger,
                    out_dir=None,
                    latent_distribution=None
                    ):
    """
    Evaluate surface distance between embedded meshes and targets.

    Args:
        deformer (torch.nn.model): Neural Flow Deformer.
        distance_metric: surface distance metric.
        test_data (Dataloader): Dataloader for the test data.
        device (str): torch device.
        logger: logger instance.
        out_dir (str): output directory.
        latent_distribution (torch.nn.model): model of the latent distribution.
    """
    cham_dist = []
    deformations = {}

    deformer.eval()
    with torch.set_grad_enabled(False):
        for ind, data_tensors in enumerate(test_mesh):  # batch size = 1
            jj = torch.tensor([data_tensors[1]], dtype=torch.long)

            # get latent
            target_latents = deformer.module.get_lat_params(jj)
            if latent_distribution:
                target_latents = latent_distribution.project_data(
                    target_latents)

            data_tensors = [
                t.unsqueeze(0).to(device) for t in data_tensors[2:]
            ]
            vi, fi, vj, fj, rescale = data_tensors
            vi = vi[0]  # batch size = 1
            fi = fi[0]  # batch size = 1
            vj = vj[0]  # batch size = 1
            fj = fj[0]  # batch size = 1

            mean_rescale = torch.mean(rescale)

            vi_j = deformer(
                vi[..., :3], target_latents,
            )

            # Check amount of deformation.
            deformation_dist_norm = vi_j[-1] - vi[..., :3]
            deformations[test_mesh.dataset.files[jj]] = deformation_dist_norm
            deform_abs = torch.mean(torch.norm(
                deformation_dist_norm, dim=-1)) * mean_rescale

            mean_rescale = mean_rescale.detach().cpu().numpy()

            # Get meshes
            src_mesh = trimesh.Trimesh(
                vi.detach().cpu().numpy()[0] * mean_rescale,
                fi.detach().cpu().numpy()[0],
                process=False
            )
            trg_mesh = trimesh.Trimesh(
                vj.detach().cpu().numpy()[0] * mean_rescale,
                fj.detach().cpu().numpy()[0],
                process=False
            )
            for a in range(len(vi_j)):
                vi_j[a] = trimesh.Trimesh(
                    vi_j[a].detach().cpu().numpy()[0] * mean_rescale,
                    fi.detach().cpu().numpy()[0],
                    process=False
                )

            # Save mesh.
            src_mesh.export(os.path.join(out_dir, f"src.obj"))
            target_file_name = os.path.basename(
                test_mesh.dataset.files[jj]).split(".")[0]
            trg_mesh.export(os.path.join(
                out_dir, f"{target_file_name}.obj"))

            for a in range(len(vi_j)):
                vi_j[a].export(os.path.join(
                    out_dir, f"{target_file_name}_deformed_layer{a}.obj"))

            vi_j[-1].export(os.path.join(
                out_dir, f"{target_file_name}_deformed.obj"))

            out_path = os.path.join(
                out_dir, f"{target_file_name}_deformed.obj")

            logger.info(f"Saved to {out_path}")

            _, level_losses = calc_loss(
                vi_j, trg_mesh, distance_metric["chamfer"], criterion)
            logging = ""
            for a, level_loss in enumerate(level_losses):
                logging += f"Level {a}:{level_loss} \t"

            cham_dist.append(level_losses[-1])

            logger.info(
                f"Test mesh number: {ind}\t"
                f"Chamfer Dist Mean: {level_losses[-1]:.6f}\t"
                f"Deform Mean: {deform_abs.item():.6f}\t"
                f"{logging}"
            )

    logger.info(
        f"Chamfer Dist Mean: {np.array(cham_dist).mean()} +/- {np.array(cham_dist).std()}\t"
    )
    return np.array(cham_dist).mean()


def test(deformer,
         device,
         test_data,
         args,
         logger,
         train_data,
         active_lod,
         epoch=60):
    """
    Applying specificity and generality script to trained model.

    Args:
        deformer (torch.nn.model): Neural Flow Deformer.
        device (str): torch device.
        test_data (Dataloader): Dataloader for the test data.
        args (dict): dict of argparse arguments.
        logger: logger instance.
        train_data (Dataloader): Dataloader for the train data.
        active_lod (int): number of trained level of details.
        epochs (int): number of epochs level of detail.
    """
    # Latent space for training shapes
    template_idx = train_data.sampler.template_idx
    lat_indices = torch.arange(deformer.module.n_shapes)
    lat_indices = lat_indices[lat_indices != template_idx]
    latents = deformer.module.get_lat_params(lat_indices)
    latent_dist = LoDPCA(latents)
    logger.info(type(latent_dist).__name__)

    # Distance metrics
    criterion = LOSSES[args.loss_type]
    chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1)
    chamfer_dist.to(device)
    chamfer_distance_metric = SurfaceDistance()
    distance_metrics = {"chamfer": chamfer_distance_metric}
    distance_loss = chamfer_dist
    loss = (distance_loss, criterion)

    # Specificity and generality
    _ = generality(deformer, device, test_data, args, logger, loss,
                   distance_metrics, latent_dist, active_lod=active_lod, epoch=epoch)
    specificity(deformer, device, train_data, args, latent_dist)


def train(args,
          deformer,
          distance_loss,
          dataloader,
          epoch,
          device,
          logger,
          optimizer
          ):
    """
    Train the model for a single epoch.

    Args:
        args (dict): dict of argparse arguments.
        deformer (torch.nn.model): Neural Flow Deformer.
        distance_loss: distance loss to be applied.
        dataloader (Dataloader): Dataloader for the training data.
        epoch (int): epoch of training.
        device (str): torch device.
        logger: logger instance.
        optimizer: torch optimizer.
    """

    tot_loss = 0
    count = 0
    criterion = LOSSES[args.loss_type]
    toc = time.time()

    deformer.train()

    for batch_idx, data_tensors in enumerate(dataloader):
        tic = time.time()
        # Send tensors to device.
        data_tensors = [t.to(device) for t in data_tensors]
        (
            _,
            jj,
            source_pts,
            target_pts,
            rescale
        ) = data_tensors

        # might want to change if scaling changes, however it only affects the logger loss
        mean_rescale = torch.mean(rescale)
        bs = len(source_pts)
        optimizer.zero_grad()

        source_points = source_pts[..., :3]
        target_latents = deformer.module.get_lat_params(jj)

        deformed_pts = deformer(
            source_points, target_latents
        )

        loss, level_losses = calc_loss(
            deformed_pts, target_pts, distance_loss, criterion)

        # backprop
        loss.backward()
        optimizer.step()

        logging = ""
        for a, level_loss in enumerate(level_losses):
            logging += "Level {}:{:.6f} \t".format(
                a, mean_rescale * level_loss)

        # Check amount of deformation.
        deform_abs = torch.mean(
            torch.norm(deformed_pts[-1] - source_pts, dim=-1)
        )

        tot_loss += level_losses[-1].detach()
        count += bs

        # Logger log.
        logger.info(
            f"Train Epoch: {epoch} [{batch_idx * bs}/{len(dataloader) * bs} ({100.0 * batch_idx / len(dataloader):.0f}%)]\t"
            f"Scaled loss: {loss.item():.6f}\t"
            f"Dist Mean: {mean_rescale * loss.item():.6f}\t"
            f"Deform Mean: {mean_rescale * deform_abs.item():.6f}\t"
            f"{logging}"
            f"DataTime: {tic - toc:.4f}\tComputeTime: {time.time() - tic:.4f}\t"

        )
        toc = time.time()

    tot_loss /= count
    return tot_loss
