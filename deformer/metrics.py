""" Implementations of surface distance metrics.

"""


import numpy as np
import trimesh
import torch

from torch import nn
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import _handle_pointcloud_input

from .definitions import REDUCTIONS


class ChamferDistKDTree(nn.Module):
    """Compute chamfer distances.

    Credits:
    Originally implemented by Chiyu Max Jiang, but enhanced with the Pytorch3D KNN.
    https://github.com/maxjiang93/ShapeFlow
    """

    def __init__(self, reduction="mean", njobs=1):
        """Initialize loss module.

        Args:
          reduction: str, reduction method. choice of mean/sum/max/min.
          njobs: int, number of parallel workers to use during eval.
        """
        super(ChamferDistKDTree, self).__init__()

        self.set_reduction_method(reduction)

    def set_reduction_method(self, reduction):
        """Set reduction method.

        Args:
          reduction: str, reduction method. choice of mean/sum/max/min.
        """
        if not (reduction in list(REDUCTIONS.keys())):
            raise ValueError(
                f"reduction method ({reduction}) not in list of "
                f"accepted values: {list(REDUCTIONS.keys())}"
            )
        self.reduce = REDUCTIONS[reduction]

    def forward(self, src, tar, target_accuracy=False):
        """
        Args:
          src: [batch, m, 3] points for source
          tar: [batch, n, 3] points for target
        Returns:
          accuracy: [batch, m], accuracy measure for each point in source
          complete: [batch, n], complete measure for each point in target
          chamfer: [batch,], chamfer distance between source and target
        """
        x, x_lengths, _ = _handle_pointcloud_input(src, None, None)
        y, y_lengths, _ = _handle_pointcloud_input(tar, None, None)

        x_nn = knn_points(x, y, lengths1=x_lengths,
                          lengths2=y_lengths, K=1, return_nn=True)
        y_nn = knn_points(y, x, lengths1=y_lengths,
                          lengths2=x_lengths, K=1, return_nn=True)

        src_to_tar_diff = (
            x_nn.knn.squeeze() - src
        )  # [b, m, 3]
        tar_to_src_diff = (
            y_nn.knn.squeeze() - tar
        )  # [b, n, 3]

        accuracy = torch.norm(src_to_tar_diff, dim=-1, keepdim=False)  # [b, m]
        complete = torch.norm(tar_to_src_diff, dim=-1, keepdim=False)  # [b, n]

        chamfer = 0.5 * (self.reduce(accuracy) + self.reduce(complete))
        if not target_accuracy:
            return chamfer  # ,accuracy, complete,
        else:
            return chamfer, self.reduce(complete), self.reduce(accuracy)


class SurfaceDistance():
    """This class calculates the symmetric vertex to surface distance of two
    trimesh meshes.
    """

    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Args:
          A: trimesh mesh
          B: trimesh mesh
        """
        _, A_B_dist, _ = trimesh.proximity.closest_point(A, B.vertices)
        _, B_A_dist, _ = trimesh.proximity.closest_point(B, A.vertices)
        distance = .5 * np.array(A_B_dist).mean() + .5 * \
            np.array(B_A_dist).mean()

        return np.array([distance])


class CorresDist(nn.Module):
    """Correspondence based average euclidean distance.
    """

    def __init__(self):
        """Initialize loss module.
        """
        super(CorresDist, self).__init__()

    def forward(self, output, target):
        """
        Args:
          output: [batch, m, 3] points for output
          target: [batch, n, 3] points for target
        """
        distance = torch.mean(torch.norm(
            output - target, dim=-1, keepdim=False), dim=-1)
        return distance
