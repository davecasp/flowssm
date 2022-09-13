from torch import nn

from pytorch3d.ops.points_alignment import corresponding_points_alignment, _apply_similarity_transform, iterative_closest_point


class ICP(nn.Module):
    def __init__(self, correspondence):
        """ ICP alignment implementation in torch.
        """
        super(ICP, self).__init__()
        self.correspondence = correspondence

    def forward(self, from_vertices, to_vertices):
        full_source = from_vertices.clone()
        start_shape = full_source.shape

        # Apply correct initial alignment
        if self.correspondence:
            R, T, s = corresponding_points_alignment(
                from_vertices,
                to_vertices,
                weights=None,
                estimate_scale=False,
                allow_reflection=False,
            )

            from_vertices = _apply_similarity_transform(from_vertices, R, T, s)

        icp = iterative_closest_point(
            from_vertices, to_vertices, relative_rmse_thr=1e-5)
        from_vertices = icp.Xt

        assert start_shape == from_vertices.shape, "Shape mismatch"
        return from_vertices
