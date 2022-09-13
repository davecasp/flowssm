"""
Credits:
Originally implemented by Chiyu Max Jiang.
https://github.com/maxjiang93/ShapeFlow

with multiple modifications and other classes.

Dataloader are to be used with .ply.
If other dataformat is needed please adjust accordingly.
"""

import glob
import numpy as np
import os
import re
import torch
import trimesh

from torch.utils.data import Dataset, Sampler, DataLoader
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes


SPLITS = ["train", "test", "val", "*", "recon"]

DATA = {
    "femur": {"n_vertices": 15000, "n_shapes": 200},
    "liver": {"n_vertices": 12000, "n_shapes": 78},
}


def rescale_center(points, extent=None, center=None):
    """
    Helper function to rescale and center meshes in [-1, 1].
    Args:
        points (numpy): point coordinates to be centered and rescaled.
        extent (float): precomputed recaling factor.
        center (numpy / tensor): offset to center.

    Returns:
        points (numpy): rescaled and centered points.
        extent (float): applied extend.
        center (tensor): applied center.
    """
    if torch.is_tensor(center):
        center = center.cpu().detach().numpy()
    if not extent:
        extent = (max(np.max(points, axis=0) - np.min(points, axis=0))) / 2
    # Scale
    points /= extent
    # Center
    if center is None:
        center = (np.max(points, axis=0) - np.min(points, axis=0)) / \
            2 - np.max(points, axis=0)
    points += center
    return points, torch.tensor(extent), torch.tensor(center)


def strip_name(filename):
    """
    Helper function to get file id from filename.
    Args:
        filename (str): Filename with ID.

    Returns:
        filename (str): extracted ID.
    """
    if len(filename.split("/")) > 1:
        filename = filename.split("/")[-1].replace(".ply", "")
    return filename


class ShapeBase(Dataset):
    """Pytorch Dataset base for loading shape data."""

    def __init__(self,
                 data_root,
                 split,
                 center_and_rescale=True,
                 template_center=False,
                 center=None,
                 extent=None):
        """
        Initialize Dataset

        Args:
            data_root (str): path to data root that contains dataset
                with train, val and test directory.
            split (str): str, one of 'train'/'val'/'test'/'*'. '*' for all splits.
            center_and_rescale (bool): whether or not to center and rescale data.
            template_center (bool): center and rescale in alignment with template shape.
            center (numpy / tensor): offset to center.
            extent (float): precomputed recaling factor.

        """
        self.data_root = data_root
        self.split = split
        self.center_and_rescale = center_and_rescale

        if not (split in SPLITS):
            raise ValueError(f"{split} must be one of {SPLITS}")

        self.files = self._get_filenames(self.data_root, self.split)
        self._file_splits = None
        self._fname_to_idx_dict = None

        self.extent = extent
        self.center = center
        if self.center_and_rescale:
            self.extent = extent
            # Compute the maximum extent if not defined
            if self.extent is None:
                self.extent = 0
                for i in range(len(self.files)):
                    vertices = trimesh.load(
                        self.files[i], process=False).vertices
                    _, extent, _ = rescale_center(vertices)
                    extent = extent.item()
                    self.extent = extent if extent > self.extent else self.extent
            if self.extent:
                print(f"Using extent of {self.extent}")
            else:
                print("Using local scaling")

            if center is None:
                if template_center:
                    self.template_file = [
                        file for file in self.files if "mean" in file]
                    vertices = trimesh.load(
                        self.template_file[0], process=False).vertices
                    _, _, center = rescale_center(vertices, extent=self.extent)

                    self.center = center.numpy()

            if self.center is not None:
                print(f"Using center of {self.center}")
            else:
                print("Using individual centering")

    @property
    def file_splits(self):
        if self._file_splits is None:
            self._file_splits = {"train": [], "test": [], "val": []}
            for f in self.files:
                if "train/" in f:
                    self._file_splits["train"].append(f)
                elif "test/" in f:
                    self._file_splits["test"].append(f)
                else:  # val/
                    self._file_splits["val"].append(f)

        return self._file_splits

    @staticmethod
    def _get_filenames(data_root, split):
        files = []
        folder = os.path.join(data_root, split)
        if not os.path.exists(folder):
            raise RuntimeError(f"Datafolder does not exist at {folder}")

        files += glob.glob(os.path.join(folder, "*.ply"), recursive=True)
        return sorted(files)

    def __len__(self):
        return self.n_shapes ** 2

    @property
    def n_shapes(self):
        return len(self.files)

    @property
    def fname_to_idx_dict(self):
        """A dict mapping unique mesh names to indicies."""
        if self._fname_to_idx_dict is None:
            fnames = [strip_name(f) for f in self.files]
            self._fname_to_idx_dict = dict(
                zip(fnames, list(range(len(fnames))))
            )
        return self._fname_to_idx_dict

    def idx_to_combinations(self, idx):
        """Converts linear index to a pair of indices."""
        i = np.floor(idx / self.n_shapes)
        j = idx - i * self.n_shapes
        if hasattr(idx, "__len__"):
            i = np.array(i, dtype=int)
            j = np.array(j, dtype=int)
        else:
            i = int(i)
            j = int(j)
        return i, j

    def combinations_to_idx(self, i, j):
        """Convert a pair of indices to a linear index."""
        idx = i * self.n_shapes + j
        if hasattr(idx, "__len__"):
            idx = np.array(idx, dtype=int)
        else:
            idx = int(idx)
        return idx


class ShapePointset(ShapeBase):
    """Pytorch Dataset for sampling surfaces points from meshes."""

    def __init__(
        self,
        data_root,
        split,
        nsamples=None,
        center_and_rescale=True,
        template_center=False,
        center=None,
        extent=None
    ):
        """
        Initialize DataSet

        Args:
            data_root (str): path to data root that contains the dataset.
            split (str): one of 'train'/'val'/'test'.
            nsamples (int): number of surface points to sample from each mesh.
            center_and_rescale (bool): whether or not to center and rescale data.
            template_center (bool): center and rescale in alignment with template shape.
            center (numpy / tensor): offset to center.
            extent (float): precomputed recaling factor.
        """
        super(ShapePointset, self).__init__(
            data_root=data_root,
            split=split,
            center_and_rescale=center_and_rescale,
            template_center=template_center,
            center=center,
            extent=extent)
        self.nsamples = nsamples

    @staticmethod
    def sample_points(mesh_path, nsamples):
        """Load the mesh from mesh_path and sample nsampels points from its surface.

        Args:
            mesh_path (str): path to load the mesh from.
            nsamples (int): number of vertices to sample.

        Returns:
          v_sample (numpy array): array of shape [nsamples, 3 or 6] for sampled points.
        """
        mesh = trimesh.load(mesh_path, process=False)
        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).float()
        mesh = Meshes(verts=[verts], faces=[faces])
        sample_points = sample_points_from_meshes(mesh, nsamples)

        sample = sample_points.numpy()
        sample = sample.squeeze()

        return sample

    def _get_one_mesh(self, idx):
        verts = self.sample_points(self.files[idx], self.nsamples)
        verts = verts
        if self.center_and_rescale:
            verts, target_scale, _ = rescale_center(
                verts, self.extent, self.center)
        else:
            target_scale = torch.Tensor([1])
        verts = torch.from_numpy(verts).float()
        return verts, target_scale

    def __getitem__(self, idx):
        """Get a random pair of shapes corresponding to idx.

        Args:
          idx (int): index of the shape pair to return.

        Returns:
            i (int): index of shape i.
            j (int): index of shape j.
            verts_i (torch tensor): [npoints, 3 or 6] float tensor for point samples from the
                first mesh.
            verts_j (torch tensor): [npoints, 3 or 6] float tensor for point samples from the
                second mesh.
            target_scale (float): applied rescaling factor.
        """
        i, j = self.idx_to_combinations(idx)
        verts_i, _ = self._get_one_mesh(i)
        verts_j, target_scale = self._get_one_mesh(j)
        return i, j, verts_i, verts_j, target_scale


class ShapeVertex(ShapeBase):
    """Pytorch Dataset for sampling vertices from meshes."""

    def __init__(
        self,
        data_root,
        split,
        nsamples=None,
        center_and_rescale=True,
        template_center=False,
        center=None,
        extent=None,
    ):
        """
        Initialize DataSet

        Args:
            data_root (str): path to data root that contains the dataset.
            split (str): one of 'train'/'val'/'test'.
            nsamples (int): number of surface points to sample from each mesh.
            center_and_rescale (bool): whether or not to center and rescale data.
            template_center (bool): center and rescale in alignment with template shape.
            center (numpy / tensor): offset to center.
            extent (float): precomputed recaling factor.
        """
        super(ShapeVertex, self).__init__(
            data_root=data_root,
            split=split,
            center_and_rescale=center_and_rescale,
            template_center=template_center,
            center=center,
            extent=extent,)
        self.nsamples = nsamples

    @staticmethod
    def sample_mesh(mesh_path, nsamples):
        """Load the mesh from mesh_path and sample nsampels points from its vertices.

        If nsamples < number of vertices on mesh, randomly repeat some
        vertices as padding.

        Args:
          mesh_path (str): path to load the mesh from.
          nsamples (int): number of vertices to sample.
        Returns:
          v (numpy array): array of shape [nsamples, 3 or 6] for sampled points.
        """
        mesh = trimesh.load(mesh_path, process=False)
        v = np.array(mesh.vertices)
        seq = np.random.permutation(len(v))[:nsamples]
        if len(seq) < nsamples:
            seq_repeat = np.random.choice(
                len(v), nsamples - len(seq), replace=True)
            seq = np.concatenate([seq, seq_repeat], axis=0)
            v = v[seq]
        return v

    def _get_one_mesh(self, idx):
        verts = self.sample_mesh(self.files[idx], self.nsamples)
        verts = verts

        if self.center_and_rescale:
            verts, target_scale, _ = rescale_center(
                verts, self.extent, self.center)
        else:
            target_scale = torch.Tensor([1])

        verts = torch.from_numpy(verts).float()
        return verts, target_scale

    def __getitem__(self, idx):
        """Get a pair of shapes corresponding to idx.

        Args:
          idx (int): index of the shape pair to return. must be smaller than
            len(self).
        Returns:
            i (int): index of shape i.
            j (int): index of shape j.
            verts_i (torch tensor): [npoints, 3 or 6] float tensor for point samples from the
                first mesh.
            verts_j (torch tensor): [npoints, 3 or 6] float tensor for point samples from the
                second mesh.
            target_scale (float): applied rescaling factor.
        """
        i, j = self.idx_to_combinations(idx)

        verts_i, _ = self._get_one_mesh(i)
        verts_j, target_scale = self._get_one_mesh(j)
        return i, j, verts_i, verts_j, target_scale


class ShapeMesh(ShapeBase):
    """Pytorch Dataset for sampling entire meshes."""

    def __init__(self,
                 data_root,
                 split,
                 center_and_rescale=True,
                 template_center=False,
                 center=None,
                 extent=None):
        """
        Initialize DataSet

        Args:
            data_root (str): path to data root that contains the dataset.
            split (str): one of 'train'/'val'/'test'.
            center_and_rescale (bool): whether or not to center and rescale data.
            template_center (bool): center and rescale in alignment with template shape.
            center (numpy / tensor): offset to center.
            extent (float): precomputed recaling factor.
        """

        super(ShapeMesh, self).__init__(
            data_root=data_root,
            split=split,
            extent=extent,
            center_and_rescale=center_and_rescale,
            template_center=template_center,
            center=center)

    def get_pairs(self, i, j):
        verts_i, faces_i, _ = self.get_single(i)
        verts_j, faces_j, target_scale = self.get_single(j)

        return i, j, verts_i, faces_i, verts_j, faces_j, target_scale

    def get_single(self, i):
        mesh_i = trimesh.load(self.files[i], process=False)

        verts_i = mesh_i.vertices
        # rescale and center
        if self.center_and_rescale:
            verts_i, target_scale, _ = rescale_center(
                verts_i, self.extent, self.center)
        else:
            target_scale = torch.Tensor([1])

        faces_i = mesh_i.faces

        verts_i = torch.from_numpy(verts_i).float()
        faces_i = torch.from_numpy(faces_i).float()

        return verts_i, faces_i, target_scale

    def __getitem__(self, idx):
        """Get a pair of meshes.
        Args:
            idx (in): index of the shape pair to return. must be smaller than
                len(self).
        Returns:
            i (int): index of shape i.
            j (int): index of shape j.
            verts_i (torch tensor): [npoints, 3 or 6] float tensor for point samples from the
                first mesh.
            faces_i (torch tensor): float tensor for mesh faces from the
                first mesh.
            verts_j (torch tensor): [npoints, 3 or 6] float tensor for point samples from the
                second mesh.
            faces_j (torch tensor): float tensor for mesh faces from the
                second mesh.
            target_scale (float): applied rescaling factor.
        """
        i, j = self.idx_to_combinations(idx)
        return self.get_pairs(i, j)


class ReconstructionMesh(ShapeBase):
    """Pytorch Dataset for reconstruction meshes (sparse and partial data)."""

    def __init__(self,
                 data_root,
                 split,
                 center_and_rescale=True,
                 template_center=False,
                 center=None,
                 extent=None,
                 sample=False):
        """
        Initialize DataSet

        Args:
            data_root (str): path to data root that contains the dataset.
            split (str): one of 'train'/'val'/'test'.
            center_and_rescale (bool): whether or not to center and rescale data.
            template_center (bool): center and rescale in alignment with template shape.
            center (numpy / tensor): offset to center.
            extent (float): precomputed recaling factor.

        """
        super(ReconstructionMesh, self).__init__(
            data_root=data_root,
            split=split,
            center_and_rescale=center_and_rescale,
            template_center=template_center,
            center=center,
            extent=extent)

        self.template_file = [file for file in self.files if "mean" in file]
        self.source_file = [file for file in self.files if (
            "_partial" in file or "_sparse" in file)]

        self.target_file = [re.sub(r"_[0-9]+_partial", "",
                                   re.sub(r"_[0-9]+_sparse", "", file)
                                   ) for file in self.source_file]

    def get_mesh(self, file, center=None):
        mesh = trimesh.load(file, process=False)

        if self.center_and_rescale:
            verts, target_scale, center = rescale_center(
                mesh.vertices, self.extent, center)
        else:
            target_scale, center = torch.Tensor([1]), torch.Tensor([0])
            verts = mesh.vertices

        verts = torch.from_numpy(verts).float()
        faces = torch.from_numpy(mesh.faces).float()

        return verts, faces, target_scale, center

    def get_points(self, file, center):
        mesh = trimesh.load(file, process=False)
        verts = mesh.vertices
        # rescale and center
        if self.center_and_rescale:
            verts, target_scale, center = rescale_center(
                mesh.vertices, self.extent, center)
        else:
            target_scale, center = torch.Tensor([1]), torch.Tensor([0])
            verts = mesh.vertices

        verts = torch.from_numpy(verts).float()
        return verts, target_scale, center

    def __len__(self):
        return len(self.source_file)

    def __getitem__(self, idx):
        """Get reconstruction data.

        Args:
          idx (int): index of the shape pair to return. must be smaller than
            len(self).

        Returns:
            idx (int): index reconstruction example.
            template_verts (torch tensor): [npoints, 3 or 6] float tensor for vertices from the
                template mesh.
            template_faces (torch tensor): float tensor for mesh faces from the
                template mesh.
            target_verts (torch tensor): [npoints, 3 or 6] float tensor for vertices from the
                target.
            target_faces (torch tensor): float tensor for mesh faces from the
                target.
            recon_verts (torch tensor): [npoints, 3 or 6] float tensor for vertices from the
                target.
            recon_faces (torch tensor): float tensor for mesh faces from the
                target if applicable (only for partial meshes).
            target_scale (float): applied rescaling factor.

        """

        target = self.target_file[idx]
        recon = self.source_file[idx]

        template_verts, template_faces, target_scale, center = self.get_mesh(
            self.template_file[0])
        target_verts, target_faces, _, _ = self.get_mesh(target, center)
        if "sparse" in recon:
            recon_verts, _, _ = self.get_points(recon, center)
            recon_faces = torch.Tensor([0])
        else:
            recon_verts, recon_faces, _, _ = self.get_mesh(recon, center)

        return idx, template_verts, template_faces, target_verts, target_faces, recon_verts, recon_faces, target_scale


class PairSamplerBase(Sampler):
    """Data sampler base for sampling pairs."""

    def __init__(
        self, dataset, src_split, tar_split, n_samples=None, replace=False
    ):
        assert src_split in SPLITS[:3]
        assert tar_split in SPLITS[:3]
        self.replace = replace
        self.n_samples = n_samples
        self.src_split = src_split
        self.tar_split = tar_split
        self.dataset = dataset
        self.src_files = self.dataset.file_splits[src_split]
        self.tar_files = self.dataset.file_splits[tar_split]
        self.src_files = [strip_name(f) for f in self.src_files]
        self.tar_files = [strip_name(f) for f in self.tar_files]

        self.n_src = len(self.src_files)
        self.n_tar = len(self.tar_files)

        if self.n_samples is None:
            self.n_samples = self.n_src - 1

        if not replace:
            if not self.n_samples <= self.n_src:
                raise RuntimeError(
                    f"Numer of samples ({self.n_samples}) must be less or equal than number source shapes ({self.n_src})"
                )

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class RandomPairSampler(PairSamplerBase):
    """Data sampler for sampling random pairs."""

    def __init__(
        self, dataset, src_split, tar_split, n_samples, replace=False
    ):
        super(RandomPairSampler, self).__init__(
            dataset, src_split, tar_split, n_samples, replace
        )

    def __iter__(self):
        d = self.dataset
        src_names = None
        tar_names = None
        while np.any(tar_names == src_names):
            src_names = np.random.permutation(self.src_files)
            tar_names = np.random.permutation(self.tar_files)
        src_idxs = np.array([d.fname_to_idx_dict[strip_name(f)]
                             for f in src_names])
        tar_idxs = np.array([d.fname_to_idx_dict[strip_name(f)]
                             for f in tar_names])
        combo_ids = self.dataset.combinations_to_idx(src_idxs, tar_idxs)

        return iter(combo_ids)

    def __len__(self):
        return self.n_samples


class TemplateShapeSampler(PairSamplerBase):
    """Data sampler for sampling template target pairs."""

    def __init__(self,
                 dataset,
                 src_split,
                 tar_split,
                 n_samples,
                 replace=False,
                 template_idx=None):
        super(TemplateShapeSampler, self).__init__(
            dataset, src_split, tar_split, n_samples, replace
        )
        """
        Initialize Sampler

        Args:
            dataset (torch dataset): dataset.
            src_split (str): template data split.
            tar_split (str): target data split.
            n_samples (int): number of pairs to be sampled.
            replace (bool): draw with raplacement
            template_idx (str): index of the template.
        """
        self.template_idx = template_idx

    def __iter__(self):
        d = self.dataset
        tar_names = None
        tar_names = np.random.permutation(self.tar_files)[:int(self.n_samples)]
        src_names = np.array([self.tar_files[self.template_idx]
                              for f in tar_names])
        while np.any(tar_names == src_names):
            tar_names = np.random.permutation(self.tar_files)[
                :int(self.n_samples)]

        src_idxs = np.array([self.template_idx for f in tar_names])
        tar_idxs = np.array([d.fname_to_idx_dict[strip_name(f)]
                             for f in tar_names])
        combo_ids = self.dataset.combinations_to_idx(src_idxs, tar_idxs)

        return iter(combo_ids)

    def __len__(self):
        return self.n_samples


def build(args,
          kwargs,
          logger,
          center_and_rescale=True,
          template_center=True):
    """
    Create datasets for neural flow deformer
    """
    if not args.sample_surface:
        logger.info("sample vertex")
        fullset = ShapeVertex(
            data_root=args.data_root,
            split="train",
            nsamples=args.samples,
            center_and_rescale=center_and_rescale,
            extent=args.global_extent,
            template_center=template_center,
            center=None
        )
    else:
        logger.info("sample points")
        fullset = ShapePointset(
            data_root=args.data_root,
            split="train",
            nsamples=args.samples,
            center_and_rescale=center_and_rescale,
            extent=args.global_extent,
            template_center=template_center,
            center=None
        )
    # Use training data for global scaling
    extent = fullset.extent

    train_meshset = ShapeMesh(
        data_root=args.data_root,
        split="train",
        center_and_rescale=center_and_rescale,
        extent=extent,
        template_center=template_center,
        center=None
    )

    # val sets for unknown embeddings
    if not args.sample_surface:
        valset = ShapeVertex(
            data_root=args.data_root,
            nsamples=args.samples,
            split="val",
            center_and_rescale=center_and_rescale,
            extent=extent,
            template_center=template_center,
            center=None
        )
    else:
        valset = ShapePointset(
            data_root=args.data_root,
            nsamples=args.samples,
            split="val",
            center_and_rescale=center_and_rescale,
            extent=extent,
            template_center=template_center,
            center=None
        )

    val_mesh_set = ShapeMesh(
        data_root=args.data_root,
        split="val",
        center_and_rescale=center_and_rescale,
        extent=extent,
        template_center=template_center,
        center=None
    )

    if not args.sample_surface:
        testset = ShapeVertex(
            data_root=args.data_root,
            nsamples=args.samples,
            split="test",
            center_and_rescale=center_and_rescale,
            extent=extent,
            template_center=template_center,
            center=None
        )
    else:
        testset = ShapePointset(
            data_root=args.data_root,
            nsamples=args.samples,
            split="test",
            center_and_rescale=center_and_rescale,
            extent=extent,
            template_center=template_center,
            center=None
        )

    test_mesh_set = ShapeMesh(
        data_root=args.data_root,
        split="test",
        center_and_rescale=center_and_rescale,
        extent=extent,
        template_center=template_center,
        center=None
    )

    # Sampler
    fname_dict = fullset.fname_to_idx_dict
    template_idx = None
    for key_ in fname_dict.keys():
        if "mean" in key_:
            template_idx = fname_dict[key_]

    # Do not draw with replacement
    replace = False
    train_sampler = TemplateShapeSampler(
        dataset=fullset,
        src_split="train",
        tar_split="train",
        n_samples=None,
        replace=replace,
        template_idx=template_idx
    )

    fname_dict = train_meshset.fname_to_idx_dict
    template_idx = None
    for key_ in fname_dict.keys():
        if "mean" in key_:
            template_idx = fname_dict[key_]

    train_mesh_sampler = TemplateShapeSampler(
        dataset=train_meshset,
        src_split="train",
        tar_split="train",
        n_samples=None,
        replace=False,
        template_idx=template_idx
    )

    if not (
        train_sampler.dataset.fname_to_idx_dict == train_meshset.fname_to_idx_dict
    ):
        raise RuntimeError(
            "Missaligned sampler between train points and meshes"
        )

    # Sampler
    fname_dict = valset.fname_to_idx_dict
    template_idx = None
    for key_ in fname_dict.keys():
        if "mean" in key_:
            template_idx = fname_dict[key_]

    val_sampler = TemplateShapeSampler(
        dataset=valset,
        src_split="val",
        tar_split="val",
        n_samples=None,
        replace=False,
        template_idx=template_idx
    )

    if not (
        val_sampler.dataset.fname_to_idx_dict == val_mesh_set.fname_to_idx_dict
    ):
        raise RuntimeError("Missmatched val sets of meshes and pointclouds")

    # Sampler
    fname_dict = testset.fname_to_idx_dict
    template_idx = None
    for key_ in fname_dict.keys():
        if "mean" in key_:
            template_idx = fname_dict[key_]

    test_sampler = TemplateShapeSampler(
        dataset=testset,
        src_split="test",
        tar_split="test",
        n_samples=None,
        replace=False,
        template_idx=template_idx
    )

    if not (
        test_sampler.dataset.fname_to_idx_dict == test_mesh_set.fname_to_idx_dict
    ):
        raise RuntimeError("Missmatched test sets of meshes and pointclouds")

    # Dataloader
    # Make sure we are turning off shuffle since we are using samplers!
    train_points = DataLoader(
        fullset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        sampler=train_sampler,
        **kwargs,
    )
    train_mesh = DataLoader(
        train_meshset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        sampler=train_mesh_sampler,
        **kwargs,
    )

    val_mesh = DataLoader(
        val_mesh_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
        **kwargs,
    )
    val_points = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
        **kwargs,
    )

    test_mesh = DataLoader(
        test_mesh_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        sampler=test_sampler,
        **kwargs,
    )
    test_points = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        sampler=test_sampler,
        **kwargs,
    )

    return train_mesh, train_points, val_mesh, val_points, test_mesh, test_points


def recon_data(args, kwargs, extent):
    """
    Get reconstruction dataloader
    """
    train_recon = ReconstructionMesh(
        args.recon_dir,
        "recon",
        center_and_rescale=True,
        extent=extent,
        template_center=True,
        center=None
    )

    mesh_recon = ReconstructionMesh(
        args.recon_dir,
        "recon",
        center_and_rescale=True,
        extent=extent,
        template_center=True,
        center=None)

    train_recon = DataLoader(
        train_recon,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    mesh_recon = DataLoader(
        mesh_recon,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    return train_recon, mesh_recon


def template_build(args, kwargs):
    """
    Create dataset for template shape registration with shapeflow.
    """

    fullset = ShapePointset(
        data_root=args.data_root,
        split="train",
        nsamples=DATA[args.data]["n_vertices"],
        center_and_rescale=True,
        extent=args.global_extent,
        template_center=False,
        center=None
    )
    extent = fullset.extent
    train_meshset = ShapeMesh(
        data_root=args.data_root,
        split="train",
        center_and_rescale=True,
        extent=extent,
        template_center=False,
        center=None
    )

    train_sampler = RandomPairSampler(
        dataset=fullset,
        src_split="train",
        tar_split="train",
        n_samples=None,
        replace=False
    )

    train_mesh_sampler = RandomPairSampler(
        dataset=train_meshset,
        src_split="train",
        tar_split="train",
        n_samples=None,
        replace=False
    )

    if not (
        train_sampler.dataset.fname_to_idx_dict == train_meshset.fname_to_idx_dict
    ):
        raise RuntimeError(
            "Missaligned sampler between train points and meshes"
        )

    train_points = DataLoader(
        fullset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        sampler=train_sampler,
        **kwargs,
    )
    train_mesh = DataLoader(
        train_meshset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        sampler=train_mesh_sampler,
        **kwargs,
    )

    return train_mesh, train_points
