import os
import random
import networkx as nx
import numpy as np
import trimesh
import point_cloud_utils as pcu


def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in zip(path_lengths.keys(), path_lengths.values()) if length <= n]


def make_sparse(file, out_path, samples=[100, 1000]):
    """
    Sample sparse point clouds from mesh.
    """
    print(file)
    for sample in samples:
        file_name = os.path.join(out_path, os.path.basename(
            file.replace(".ply", f"_{sample}_sparse.ply")))
        if not os.path.exists(file_name):
            v, f, _ = pcu.load_mesh_vfn(file)
            m = 0
            tolerance = 0.95
            i = 0
            while abs(m/sample) < tolerance:
                f_i, bc = pcu.sample_mesh_poisson_disk(
                    v, f, sample)
                grid = pcu.interpolate_barycentric_coords(
                    f, f_i, bc, v)
                m = grid.shape[0]
                i += 1
                print(abs(m/sample))

            trimesh.points.PointCloud(grid).export(file_name)


def make_partial(file, out_path, hops=20):
    """
    Get partial meshes by removing vertex and its neighborhood.
    """
    mesh = trimesh.load(file, process=False)
    n_verts = mesh.vertices.shape[0]
    file_name = os.path.basename(file)
    mesh = trimesh.load(file, process=False)
    mesh.export(os.path.join(out_path, file_name))

    for i in random.sample(range(0, n_verts), 2):
        mesh = trimesh.load(file, process=False)
        G = nx.Graph()
        G.add_edges_from(mesh.edges)
        idxs = neighborhood(G, i, hops)
        mask_vertices = np.zeros(len(mesh.vertices), dtype=np.bool)
        mask_vertices[idxs] = True
        mask_vertices = np.invert(mask_vertices)
        mask = mask_vertices[mesh.faces].any(axis=1)

        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()

        file_name = os.path.basename(file.replace(".ply", f"_{i}_partial.ply"))
        print(file_name)
        mesh.export(os.path.join(out_path, file_name))
