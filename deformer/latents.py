""" Models to estimate latent representations.
"""
import torch
import torch.nn as nn

from deformer.pca import PCA


class LatentDist(nn.Module):
    def __init__(self):
        super(LatentDist, self).__init__()

    def flatten(self, data):
        self.shape = [level.shape for level in data]
        data = torch.cat([level.flatten(start_dim=1) for level in data], dim=1)
        return data

    def unflatten(self, data, n_samples=None):
        """ Only works for grid latents.
        """
        i = 0
        latent = []
        for shape in self.shape:
            if n_samples:
                shape = list(shape)
                shape[0] = n_samples
                shape = tuple(shape)
            length = shape[-1]**3 * shape[1]
            latent.append(data[:, i: i + length].reshape(shape))
            i += length

        return latent

    def sample_list(self, data):
        """ Only works for lod grid latents.
        """
        unflattened_lat = []
        for i in range(data[0].shape[0]):
            latent_level_list = []
            for lod in data:
                latent_level_list.append(lod[i].unsqueeze(0))
            unflattened_lat.append(latent_level_list)

        return unflattened_lat

    def project_data(self, data):
        """Identity mapping"""
        return data


class LoDPCA(LatentDist):
    def __init__(self, data):
        """LoD PCA wrapper.

        Args:
            data (tensor): data tensor with k examples, each n-dimentional: k x n matrix
        """
        super(LoDPCA, self).__init__()
        self.pcas = torch.nn.ModuleList([PCA(level) for level in data])

    def project_data(self, data, num_modes=None):
        """Project a lod data into the PCA.

        Args:
            data (tensor): data tensor with k examples, each n-dimentional: k x n matrix
            num_modes (int): number of modes.
        """
        data_proj = [pca.project_data(level, num_modes)
                     for level, pca in zip(data, self.pcas)]
        return data_proj

    def generate_random_samples(self, nsamples: int = 1, num_modes=None):
        """Generate random samples from distribution

        Args:
            nsamples (int): number of samples.
            num_modes (int): number of modes.
        """
        samples = [pca.generate_random_samples(
            nsamples, num_modes) for pca in self.pcas]
        return self.sample_list(samples)

    def __len__(self):
        return len(self.pcas[-1])

    def forward(self, weights, num_modes=None):
        data_proj = [pca(weigth, num_modes)
                     for weigth, pca in zip(weights, self.pcas)]
        return data_proj

    def get_weights(self, data):
        weights = [pca.get_weigths(dat) for dat, pca in zip(data, self.pcas)]
        return weights


class GlobalPCA(LatentDist):
    def __init__(self, data):
        """Global PCA wrapper.

        :param data: data tensor with k examples, each n-dimentional: k x n matrix
        """
        super(GlobalPCA, self).__init__()
        data = self.flatten(data)
        self.pca = PCA(data)

    def project_data(self, data, num_modes=None):
        """Project a lod data into the PCA."""
        data = self.flatten(data)
        data_proj = self.pca.project_data(data, num_modes)
        return self.unflatten(data_proj)

    def generate_random_samples(self, nsamples: int = 1, num_modes=None):
        data = self.pca.generate_random_samples(
            nsamples, num_modes)
        return self.sample_list(self.unflatten(data, nsamples))

    def __len__(self):
        return len(self.pca)

    def forward(self, weights, num_modes=None):
        data_proj = self.pca(weights, num_modes)
        return data_proj


class LatentGaussDist(LatentDist):
    def __init__(self, data):
        """Gaussian distribution sampler for latent.

        :param data: data tensor with k examples, each n-dimentional: k x n matrix
        """
        super(LatentGaussDist, self).__init__()
        with torch.no_grad():
            data = self.flatten(data)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            self.normal = torch.distributions.normal.Normal(
                self.mean, self.std)
            self.len = len(self.mean)

    def generate_random_samples(self, nsamples=1, num_modes=None):
        """Sample latents from indipendent Gaussians."""
        with torch.no_grad():
            data = self.normal.sample((1, nsamples)).squeeze()
            return self.sample_list(self.unflatten(data, nsamples))

    def project_data(self, data, num_modes=None):
        """Identity, i.e. no projection"""
        return data

    def __len__(self):
        return len(self.mean)


class IIDLatentGaussDist(LatentDist):
    def __init__(self, data, std=0.1):
        """IID Gaussian distribution sampler for latent.
        """
        super(IIDLatentGaussDist, self).__init__()
        with torch.no_grad():
            data = self.flatten(data)
            # Initialize iid Gaussian
            self.mean = torch.zeros(data.shape[1])
            self.std = torch.ones(data.shape[1]) * std
            self.normal = torch.distributions.normal.Normal(
                self.mean.detach(), self.std.detach())
            self.len = len(self.mean)

    def generate_random_samples(self, nsamples=1, num_modes=None):
        """Sample latents from iid Gaussians."""
        with torch.no_grad():
            data = self.normal.sample((1, nsamples)).squeeze()
            return self.sample_list(self.unflatten(data, nsamples))

    def project_data(self, data, num_modes=None):
        """Identity, i.e. no projection"""
        return data

    def __len__(self):
        return len(self.mean)
