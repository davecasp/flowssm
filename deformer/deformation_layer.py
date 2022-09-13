"""
Credits:
Initial implementation by Chiyu Max Jiang
https://github.com/maxjiang93/ShapeFlow

New functionality, classes and functions.
"""

import torch
from torch import nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_regular

from utils.dataloader import rescale_center
from .definitions import NONLINEARITIES
import numpy as np
import point_cloud_utils as pcu

# Pytorch does not provide a pi implementation (calculate an approximation)
torch.pi = torch.acos(torch.zeros(1)).item() * 2


class RadialBasisFunctionSampler3D(nn.Module):
    """Radial Basis Function interpolation for irregular and regular grid.
    """

    def __init__(self,
                 irregular_path=None,
                 n_gridpoints=8,
                 extent=None,
                 independent_epsilon=True,
                 epsilon=0.5) -> None:
        """Initialization.

        Args:
            irregular_path (str): path to template to sample irregular grid points
                            from (regular grid in R^3 from -1.0 to 1.0 if None).
            n_gridpoints (int): total number of gridpoints to be cubed.
            extent: float, max extent of shape to rescale template.
            independent_epsilon: bool, independent epsilon per point.
            epsilon: float, epsilon to be multiplied by n_gridpoints.
        """
        super(RadialBasisFunctionSampler3D, self).__init__()
        self.grid_dim = n_gridpoints
        epsilon = epsilon * n_gridpoints
        n_gridpoints = n_gridpoints**3

        # Initialize epsilon(s)
        if independent_epsilon:
            # independent parameter to be tuned
            self.n_epsilon = n_gridpoints
            self.epsilon = torch.nn.Parameter(
                torch.ones((self.n_epsilon)) * epsilon)
        else:
            self.n_epsilon = 1
            epsilon = torch.ones((self.n_epsilon)) * epsilon
            self.register_buffer('epsilon', epsilon)

        self.get_controll_points(n_gridpoints, irregular_path, extent)
        self.interpolation = n_gridpoints != 1
        self.n_grid = len(self.grid)

    def forward(self, points, latents):
        """Compute radial basis function interpolation.

        Args:
            points (tensor): Float tensor of shape [batch, num_points, dim]
            latents (tensor): Float latent parameter tensor of shape [batch, latent_size, grid_dim, grid_dim, grid_dim]

        Returns:
            latent (tensor): Interpolated latent parameters
        """
        B, N, _ = points.shape
        latents = latents.reshape((B, latents.shape[-1]**3, latents.shape[1]))
        latents = latents[:, :self.n_grid, :]
        if not self.interpolation:
            return latents.repeat((1, N, 1))
        else:
            with torch.no_grad():
                dist = self.distance(points)
            data = self.rb_function(dist)
            return torch.bmm(data, latents)

    def distance(self, points):
        """Calculate point to point distance between points and grid.

        Args:
            points (tensor): Float tensor of shape [batch, num_points, dim]

        Returns:
            dist (tensor): Float tensor of the pair-wise distances [batch, num_points, n_grid]
        """
        return torch.cdist(points, self.grid)

    def rb_function(self, data):
        """Gaussian radial basis function kernel.

        Args:
            data (tensor): Float tensor of distance data of shape [batch, num_points, n_grid]
        """
        return torch.exp(-((self.epsilon * data)**2))

    def get_controll_points(self, n_gridpoints, irregular_path, extent):
        """Get regular grid or irregular controll points for RBF interpolation.
        If irregular_path is not path to template then set regular grid.

        Args:
            n_gridpoints (int): total number of gridpoints.
            irregular_path (str): path to template to sample irregular grid points
                            from.
            extent: float, max extent of shape to rescale template.
        """

        irregular = isinstance(irregular_path, str)

        with torch.no_grad():
            if not irregular:
                # get regular grid with n grid points
                print("regular grid ", n_gridpoints)
                if n_gridpoints == 1:
                    points = torch.tensor([0])
                else:
                    points = torch.linspace(-1, 1, self.grid_dim)
                x, y, z = torch.meshgrid(points, points, points, indexing="ij")
                grid = torch.cat((x.flatten()[:, None], y.flatten()[:, None],
                                  z.flatten()[:, None]), dim=1).float()  # [grid_dim, 3]
            else:
                # Get irregular grid
                if n_gridpoints == 1:
                    points = torch.tensor([0])
                    x, y, z = torch.meshgrid(
                        points, points, points, indexing="ij")
                    grid = torch.cat((x.flatten()[:, None], y.flatten()[
                                     :, None], z.flatten()[:, None]), dim=1).float()
                else:
                    v, f, n = pcu.load_mesh_vfn(irregular_path)
                    v, _, _ = rescale_center(v, extent)
                    n = 0
                    i = 0
                    n_max = 0
                    max_grid = None
                    # The poisson disk sampling does not always returns n points.
                    # Rerun if necessary
                    while n != n_gridpoints and i < 100000:
                        # Use poisson disk sampling for irregular grid
                        f_i, bc = pcu.sample_mesh_poisson_disk(
                            v, f, n_gridpoints)
                        grid = pcu.interpolate_barycentric_coords(
                            f, f_i, bc, v)

                        n = grid.shape[0]
                        i += 1
                        if n > n_max and n <= n_gridpoints:
                            n_max = n
                            max_grid = grid

                    grid = torch.from_numpy(max_grid).float()

        # might want to set grid point to be optimized
        # self.grid = torch.nn.Parameter(grid)

        # Register grid points as buffer
        self.register_buffer('grid', grid)

    def set_epsilon(self, epsilon):
        """Set epsilon value.
        """
        self.epsilon = torch.nn.Parameter(torch.ones((self.n_epsilon),
                                                     device=self.epsilon.device) *
                                          epsilon * self.grid_dim)


class TrilinearInterpolation(nn.Module):
    """
    Trilinear interpolation for regular grid.
    """

    def __init__(self,
                 ) -> None:
        """
        Initialization.
        """
        super(TrilinearInterpolation, self).__init__()

    def forward(self, points, latents):
        """
        Compute trilinear interpolation.
        """
        B, N, _ = points.shape
        points_sample = points.reshape(B, N, 1, 1, 3)
        return torch.nn.functional.grid_sample(latents,
                                               points_sample,
                                               mode="bilinear",
                                               padding_mode='border',
                                               align_corners=True,
                                               )[:, :, :, 0, 0].transpose(1, 2)


class StationaryVelocityGridField(nn.Module):
    """
    Implementation of a stationary velocity field, 
    which is computed by trilinear interpolation of grid points.
    """

    def __init__(
        self,
    ):
        super(StationaryVelocityGridField, self).__init__()
        self.lat_params = None
        # Set default interpolation
        self.interpolator = TrilinearInterpolation()

    def update_latents(self, latents):
        self.latent_sequence = latents
        self.latent_updated = True

    def add_rbf(self, rbf):
        # change default interpolator
        self.interpolator = rbf

    def get_rbf(self):
        Warning(f"get_rbf is not implemented for {self.__name__}")
        pass

    def add_lat_params(self, lat_params):
        self.lat_params = lat_params

    def add_fourier(self, A, B):
        pass

    def get_lat_params(self, idx):
        assert self.lat_params is not None
        return self.lat_params[idx]

    def forward(self, t, points):
        assert self.latent_updated, "Latent is not set"
        velocity = self.interpolator(points, self.latent_sequence)

        return velocity


class ImNet(nn.Module):
    """
    ImNet MLP parameterizing the neural ODE.
    """

    def __init__(
        self,
        dim=3,
        in_features=32,
        out_features=4,
        nf=32,
        nonlinearity="leakyrelu",
    ):
        """Initialization.

        Args:
            dim (int): dimension of input points.
            in_features (int): length of input features (i.e., latent code).
            out_features (int): number of output features.
            nf (int): width of the second to last layer.
            nonlinearity (str): torch nonlinearity.
        """
        super(ImNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.activ = NONLINEARITIES[nonlinearity]
        self.fc0 = nn.Linear(self.dimz, nf * 16)
        self.fc1 = nn.Linear(nf * 16 + self.dimz, nf * 8)
        self.fc2 = nn.Linear(nf * 8 + self.dimz, nf * 4)
        self.fc3 = nn.Linear(nf * 4 + self.dimz, nf * 2)
        self.fc4 = nn.Linear(nf * 2 + self.dimz, nf * 1)
        self.fc5 = nn.Linear(nf * 1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):
        """Forward pass MLP.

        Args:
            x (tensor): [batch_size, dim+in_features] tensor, inputs to decode.
        Returns:
            output (tensor): output of shape [batch_size, out_features].
        """
        x_tmp = x
        for dense in self.fc[:4]:
            x_tmp = self.activ(dense(x_tmp))
            x_tmp = torch.cat([x_tmp, x], dim=-1)
        x_tmp = self.activ(self.fc4(x_tmp))
        x_tmp = self.fc5(x_tmp)
        return x_tmp


class NeuralFlowModel(nn.Module):
    """
    Level of detail neural flow deformer parameterized by ImNet MLP.
    """

    def __init__(
        self,
        dim=3,
        out_features=3,
        latent_size=1,
        f_width=50,
        nonlinearity="relu",
    ):
        """Initialization.

        Args:
            dim (int): dimension of input points.
            out_features (int): number of output features.
            in_features (int): length of input features (i.e., latent code).
            latent_size(int): size of latent space. >= 1.
            f_width (int): width of ImNet.
            nonlinearity (str): torch nonlinearity.
        """
        super(NeuralFlowModel, self).__init__()

        self.out_features = out_features
        self.fourier = False

        self.net = ImNet(
            dim=dim,
            in_features=latent_size,
            out_features=out_features,
            nf=f_width,
            nonlinearity=nonlinearity,
        )

        self.latent_updated = False
        self.rbf = None
        self.lat_params = None
        self.scale = nn.Parameter(torch.ones(1) * 1e-3)

        # Initialize weight of ImNet
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def add_fourier(self, A, B):
        """
        Add fourier transform parameters.
        """
        self.flow_net.A = A
        self.flow_net.B = B
        self.flow_net.fourier = True

    def add_lat_params(self, lat_params):
        """
        Add latent parameters.
        """
        self.lat_params = lat_params

    def add_rbf(self, rbf):
        """
        Add radial basis function interpolation.
        """
        self.rbf = rbf

    def get_lat_params(self, idx):
        """
        Get latent parameters for indices.
        """
        assert self.lat_params is not None
        return self.lat_params[idx]

    def get_rbf(self):
        """
        Get radial basis function interpolator.
        """
        return self.rbf

    def fourier_encode(self, points):
        """
        Apply fourier feature encoding on points based on 
        https://bmild.github.io/fourfeat/.
        """
        try:
            features = [self.A * torch.sin((2. * torch.pi * points) @ self.B.T),
                        self.A * torch.cos((2. * torch.pi * points) @ self.B.T)]
            return torch.cat(features, dim=-1) / torch.linalg.norm(self.A)
        except Exception:
            print("failed fourier")
            points

    def update_latents(self, latent_sequence):
        """
        Save latents for ODEINT solve and compute vectornorm on latent.
        """
        self.latent_sequence = latent_sequence
        self.latent_norm = torch.norm(
            self.latent_sequence, dim=-1
        )
        self.latent_updated = True

    def latent_at_t(self, t):
        """
        Helper function to compute latent at t.
        """
        latent_val = t * self.latent_sequence  # [batch, latent_size]
        return latent_val

    def get_velocity(self, latent_vector, points):
        """
        Compute velocity at points given their latent representation.

        Args:
            latent_vector (tensor): tensor of shape [batch, latent_size], latent code for
                each shape
            points (tensor): tensor of shape [batch, num_points, dim], points representing
                each shape

        Returns:
            out (tensor): tensor of shape [batch, num_points, dim], velocity at
                each point
        """
        if self.fourier:
            points = self.fourier_encode(points)
            # [batch, num_points, latent_size]
            points_latents = points + latent_vector

        else:
            # [batch, num_points, dim + latent_size]
            points_latents = torch.cat((points, latent_vector), axis=-1)

        b, n, d = points_latents.shape
        out = self.net(points_latents.reshape([-1, d]))
        out = out.reshape([b, n, self.out_features])
        return out

    def forward(self, t, points):
        """
        Forward pass to compute instantaneous velocity of the flow at point and time.

        Args:
            t (float): deformation parameter between 0 and 1.
            points (tensor): point coordinates [batch, num_points, dim]
        Returns:
            velocity (tensor): [batch, num_points, dim]
        """
        if not self.latent_updated:
            raise RuntimeError(
                "Latent not updated. "
                "Use .update_latents() to update the source and target latents"
            )

        t = t.to(self.latent_sequence.device)

        # Reparametrize along latent path as a function of a single
        # scalar t
        latent_val = self.latent_at_t(t)

        # Compute velocity
        velocity = self.get_velocity(
            latent_val, points)  # [batch, num_pints, dim]
        # Normalize velocity based on latent vector norm
        velocity *= self.latent_norm[:, :, None]
        return velocity * self.scale


class NeuralFlowDeformer(nn.Module):
    """
    Wrapper of level of detail neural flow deformer. 
    This class manages the level of details and odeint solver.
    """

    def __init__(
        self,
        dim=3,
        out_features=3,
        latent_size=1,
        f_width=50,
        method="dopri5",
        nonlinearity="leakyrelu",
        adjoint=True,
        atol=1e-5,
        rtol=1e-5,
        lod=None,
        at_layer=0,
        rbf=None,
        grid_vector_field=False,
    ):
        """
        Initialize.

        Args:
            dim (int): physical dimensions. Either 2 for 2d or 3 for 3d.
            out_features (int): number of output features.
            latent_size (int): size of latent space. >= 1.
            f_width (int): number of neurons per hidden layer for flow network
                (>= 1).
            adjoint (bool): whether to use adjoint solver to backprop gadient
                thru odeint.
            rtol, atol (float): relative / absolute error tolerence in ode solver.
            method (str): method to solve ODE.
            nonlinearity (str): torch nonlinearity.
            lod (list): indicating the dimensionality of the level of details.
            at_layer (int): current layer being trained.
            rbf (tuple): with instanciation values for rbf interpolation.
            grid_vector_field (bool): whether to use a velocity field defined on a grid or MLP.
        """
        super(NeuralFlowDeformer, self).__init__()

        self.grid_vector_field = grid_vector_field
        self.lod = lod
        self.at_layer = at_layer

        # ODEINT
        self.method = method
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol
        self.odeints = [
            odeint_adjoint if adjoint else odeint_regular] * len(self.lod)
        self.__timing = torch.from_numpy(
            np.array([0.0, 1.0]).astype("float32")
        ).float()

        # Neural Flow Model
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        self.f_width = f_width
        self.rbf = rbf
        self.latent_size = latent_size

        self.net = torch.nn.ModuleList([])
        for i in range(len(self.lod)):
            if grid_vector_field:
                self.net.append(StationaryVelocityGridField())
            else:
                self.net.append(NeuralFlowModel(
                    dim=dim,
                    out_features=out_features,
                    latent_size=latent_size,
                    f_width=f_width,
                    nonlinearity=nonlinearity
                ))
        if rbf:
            (irregular_path, extent, independent_epsilon,
                epsilon, n_grid_p) = rbf
            for i in range(len(self.lod)):
                self.net[i].add_rbf(RadialBasisFunctionSampler3D(irregular_path=irregular_path,
                                                                 n_gridpoints=n_grid_p[i],
                                                                 extent=extent,
                                                                 independent_epsilon=independent_epsilon,
                                                                 epsilon=epsilon))
        else:
            self.interpolator = TrilinearInterpolation()
        self.lasts = [parameter.data for name,
                      parameter in self.net[0].named_parameters() if "fc5" in name]

        # init gradients for layers
        for i in range(min(self.at_layer, len(self.net))):
            self.set_latent_gradient(i)
            self.set_layer_gradient(i)

    @ property
    def adjoint(self):
        return self.__adjoint

    @ adjoint.setter
    def adjoint(self, isadjoint):
        assert isinstance(isadjoint, bool)
        self.__adjoint = isadjoint

    @ property
    def timing(self):
        return self.__timing

    @ property
    def n_shapes(self):
        n = self.net[0].lat_params.shape[0]
        return n

    @ timing.setter
    def timing(self, timing):
        assert isinstance(timing, torch.tensor)
        assert timing.ndim == 1
        self.__timing = timing

    def add_lat_params(self, lat_params):
        """
        Add latent parameters.
        """
        for i in range(len(self.lod)):
            self.net[i].add_lat_params(lat_params[i])

    def add_fourier(self, A, B):
        """
        Add fourier transform parameters.
        """
        for i in range(len(self.lod)):
            self.net[i].add_fourier(A, B)

    def get_lat_params(self, idx):
        """
        Get lod latent parameters for indices.
        """
        latents = []
        for i in range(len(self.lod)):
            latents.append(self.net[i].get_lat_params(idx))
        return latents

    def set_at_layer(self, layer_id):
        self.at_layer = layer_id

    def set_latent_gradient(self, idx, freeze=False):
        """
        Set requires_grad to freeze for latent parameters of lod.
        """
        for name, param in self.net[idx].named_parameters():
            if 'lat' in name:
                param.requires_grad = freeze

    def set_layer_gradient(self, idx, freeze=False):
        """
        Set requires_grad to freeze for network hyperparameter of lod.
        """
        for name, param in self.net[idx].named_parameters():
            if 'lat' not in name:
                param.requires_grad = freeze

    def __str__(self):
        if self.grid_vector_field:
            return ""

        string = "\n\n################################################################################################\n"
        string += "######################################### Model Summary ########################################\n"
        string += "################################################################################################\n"
        string += "\n\nPyTorch implementation of a continuous flow-field deformer.\n\n"
        string += "------------------------------------------------------------------------------------------------\n"
        string += "-------------------------------------------Model set-up-----------------------------------------\n"
        string += "------------------------------------------------------------------------------------------------\n"
        string += f"{'3D' if self.out_features == 3 else '2D'} imnet deformers without sign network, that "
        string += f"applies template deformations\n"

        string += f"It leverages {len(self.net)} sequential deformation(s), with lod resolution(s) of {['{}x{}x{}'.format(l.shape[3],l.shape[3],l.shape[3]) for l in self.get_lat_params(0)]}\n"

        string += "------------------------------------------------------------------------------------------------\n"
        string += "--------------------------------------------MLP set-up------------------------------------------\n"
        string += "------------------------------------------------------------------------------------------------\n"

        string += f"6 MLP layers with width of {self.f_width}\n"
        string += f"{self.latent_size} latent features\n"
        string += f"{self.nonlinearity} activation function\n"
        string += "------------------------------------------------------------------------------------------------\n"
        string += "---------------------------------------------Odeint---------------------------------------------\n"
        string += "------------------------------------------------------------------------------------------------\n"
        string += f"{self.method} odeint method {'with' if self.adjoint else 'without'} adjoint (absolute tolerance {self.atol}, relative tolerance {self.rtol})\n\n"

        string += "------------------------------------------------------------------------------------------------\n"
        string += "-------------------------------------------Parameters-------------------------------------------\n"
        string += "------------------------------------------------------------------------------------------------\n"

        for n, deformer in enumerate(self.net):
            string += f"Deformer {n} with {sum(p.numel() for name, p in deformer.named_parameters() if p.requires_grad and 'flow_net' in name)} trainable ({sum(p.numel() for name, p in deformer.named_parameters() if 'flow_net' in name)}) parameters\n"
            string += f"Deformer {n} with {sum(p.numel() for name, p in deformer.named_parameters() if p.requires_grad and 'lat_params' in name)} trainable ({sum(p.numel() for name, p in deformer.named_parameters() if 'lat_params' in name)}) latent features\n"
        if self.rbf:
            for net in self.net:
                string += f"RBF epsilon of {net.get_rbf().epsilon}.\n"

        string += "################################################################################################\n"
        string += "################################################################################################\n"
        string += "\n\n"

        return string

    def forward(self, points, latent_sequence):
        """
        Transformation source -> target.

        Args:
            points (tensor): Float tensor of shape [batch, num_points, dim]
            latent_sequence (list): list of float tensor of shape [batch, latent_size],

        Returns:
            points_deformed (list):
                list of tensors of shape [batch, num_points, dim] 
                with positions after each LOD.
        """
        timing = self.timing.to(points.device)
        deformations = []
        targets = []

        points_sample = points

        # Iterate over level of detail
        for i in range(len(latent_sequence)):

            if self.grid_vector_field:
                ########################################
                ## Discrete stationary velocity field ##
                ########################################

                # set velocity field parameterized with latent
                _ = self.net[i].update_latents(latent_sequence[i])

                points_deformed = self.odeints[i](
                    self.net[i],
                    points,
                    timing,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                deformations.append(points_deformed[-1])

                points = points_deformed[-1]

                if i == self.at_layer:
                    break
            else:
                ##################
                ## MLP DEFORMER ##
                ##################

                # Sample z from lod
                if not self.rbf:
                    # Sample latents with trilinear interpolation
                    targets.append(self.interpolator(points_sample,
                                                     latent_sequence[i]
                                                     ))
                else:
                    # Sample latents with radial basis function interpolation
                    targets.append(self.net[i].get_rbf()(
                        points_sample,
                        latent_sequence[i]
                    ))

                target = targets[-1]
                _latent_sequence = target
                _ = self.net[i].update_latents(_latent_sequence)

                # solve ODEINT
                points_deformed = self.odeints[i](
                    self.net[i],
                    points,
                    timing,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                deformations.append(points_deformed[-1])
                points = points_deformed[-1]

                if i == self.at_layer:
                    break
        return deformations
