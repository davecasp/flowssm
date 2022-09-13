"""
Evaluate generality and specificity for different models.

Note: The data from each model has followed different naming conventions. 
"""

import argparse
import glob
import json
import multiprocessing as mp
import numpy as np
import os
import torch
import trimesh
import pymeshlab

from tqdm import tqdm

import utils.dataloader as dl
from deformer.metrics import ChamferDistKDTree, SurfaceDistance
from utils import utility_functions as utils
from utils.icp import ICP


def get_parser():
    parser = argparse.ArgumentParser(
        description="Experimental evaluation on generated meshes")
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to directory including sampled meshes and generality meshes."
    )
    parser.add_argument(
        "--gt_data",
        type=str,
        help="Path to ground truth data."
    )
    args = parser.parse_args()
    return args


def self_intersections(file):
    """Helperfunction computing number of self-intersections.

    Args:
        file (str): filepath.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file)
    ms.apply_filter('select_self_intersecting_faces')
    m = ms.current_mesh()
    n_intersections = m.selected_face_number()
    # clear
    ms.clear()
    del ms
    del m
    return n_intersections


def intersection(mesh_files):
    """
    Compute number of selfintersections for mesh_files.

    Args:
        mesh_files (list): list of mesh file paths.
    """
    intersecetions = {}

    pool = mp.Pool(mp.cpu_count())

    results = [result for result in pool.map(
        self_intersections, mesh_files)]

    n_meshes_with_self_intersections = int(
        np.sum(np.array(results) > 0))

    intersecetions["median_faces"] = np.median(
        np.array(results))
    intersecetions["n_shapes"] = n_meshes_with_self_intersections

    intersecetions["mean_faces"] = np.array(
        results).sum()/n_meshes_with_self_intersections

    pool.close()
    pool.join()

    print(intersecetions)

    return intersecetions


def specificity(sample_mesh_files, train_data, device, logger, iterative_closest_points):
    """
    Compute specificity error.

    Args:
        sample_mesh_files (list): list of filenames.
        train_data (torch dataloader): dataloader for training data
        device (str): torch device.
        logger: logger instance
        iterative_closest_points (torch.nn.Model): ICP for shape alignment.
    """
    with torch.no_grad():
        # get chamfer distance instance
        chamfer_dist = ChamferDistKDTree(reduction="mean")
        chamfer_dist.to(device)

        sampled_mesh_vertices = []
        # get 1000 samples for evaluation
        for i, mesh_file in enumerate(sample_mesh_files):
            mesh = trimesh.load(mesh_file, process=False)
            sampled_mesh_vertices.append(torch.from_numpy(
                mesh.vertices).float().unsqueeze(dim=0))

        sampled_mesh_vertices = torch.cat(
            sampled_mesh_vertices).to(device)

        distances = np.zeros(
            (len(sampled_mesh_vertices), len(train_data)))

        # Check for closest training shape
        # Batch size 1
        for i, data_tensors in enumerate(tqdm(train_data)):
            (_, _, _, _, target_pts, _, rescale) = data_tensors

            scaled_vertices = target_pts.to(
                device) * rescale.to(device)

            # align
            aligned_vertices = iterative_closest_points(scaled_vertices.repeat(
                len(sampled_mesh_vertices), 1, 1), sampled_mesh_vertices)

            # calculate the distances
            dist = chamfer_dist(
                aligned_vertices, sampled_mesh_vertices).detach().cpu().numpy()
            distances[:, i] = dist

        # get minimum distance to any of the training shapes
        chamfer = distances.min(1).mean()
        chamfer_std = distances.min(1).std()

    print(chamfer, chamfer_std)
    return (chamfer, chamfer_std)


def generality(embedded_mesh_files, iterative_closest_points, device, target_folder):
    """Compute surface distance of embedded shapes to ground truth targets.

    Args:
        embedded_mesh_files (list): list of file paths of the embedded shape.
        iterative_closest_points (torch.nn.Model): ICP for shape alignment.
        device (str): torch device.
        target_folder (str): directory path of the ground truth targets.
    """
    with torch.no_grad():
        chamfer_distance_metric = SurfaceDistance()
        targets = []

        # generate target file paths
        for file in embedded_mesh_files:
            file = os.path.basename(file)
            case_id = "".join(filter(str.isdigit, file))  # [:-1]
            file = glob.glob(os.path.join(target_folder, f"*/*{case_id}*"))
            if len(file) == 1:
                targets.append(file[0])

        # load meshes
        def_mesh = [trimesh.load(deformed, process=False)
                    for deformed in embedded_mesh_files]
        target_mesh = [trimesh.load(target, process=False)
                       for target in targets]

        # Align shapes
        aligned_deformed_mesh = []
        for (mesh_source, mesh_target) in zip(def_mesh, target_mesh):
            vertices_source = torch.from_numpy(
                mesh_source.vertices).unsqueeze(0).to(device).float()
            vertices_target = torch.from_numpy(
                mesh_target.vertices).unsqueeze(0).to(device).float()
            vertices_source = iterative_closest_points(
                vertices_source, vertices_target)
            mesh_source.vertices = vertices_source.cpu().numpy()[0]

            aligned_deformed_mesh.append(mesh_source)

        data = zip(aligned_deformed_mesh, target_mesh)

        # multi process
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(chamfer_distance_metric, data)
        distances = np.concatenate(results)

        surface_dist = distances.mean()
        surface_dist_std = distances.std()

        pool.close()
        pool.join()

        print(surface_dist, surface_dist_std)

        return (surface_dist, surface_dist_std)


def main():
    """Evaluation of the Neural Flow Deformer.
    """
    args = get_parser()

    with open(os.path.join(args.directory, 'params.json')) as f:
        arg_dict = json.load(f)
        data_args = argparse.Namespace(**arg_dict)

    logger = utils.get_logger(log_dir=args.directory)

    # calculate the maximum extent for scaling
    data_args.global_extent = None if data_args.global_extent else data_args.global_extent

    use_cuda = torch.cuda.is_available()
    kwargs = (
        {"num_workers": min(6, data_args.batch_size), "pin_memory": True}
        if use_cuda
        else {}
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    # Deformer is landmark free so no correspondence intitialization
    iterative_closest_points = ICP(False)
    iterative_closest_points.to(device)

    logger.info(data_args)

    # get data
    train_mesh, _, _, _, _, _ = dl.build(
        data_args, kwargs, center_and_rescale=True, template_center=True, logger=logger)

    generality_files = glob.glob(os.path.join(
        args.directory, "generality_meshes/*deformed.obj"))
    specificity_files = glob.glob(
        os.path.join(args.directory, "sampled_meshes/*.ply"))

    # compute
    _ = intersection(generality_files)
    _ = intersection(specificity_files)

    _ = generality(generality_files,
                   iterative_closest_points, device, args.gt_data)
    _ = specificity(specificity_files, train_mesh, device,
                    logger, iterative_closest_points)


if __name__ == "__main__":
    main()
