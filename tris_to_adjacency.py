import numpy as np
from scipy import sparse


def tris_to_adjacency(trimesh, n):
    """Function to transform trimeshs from grid to adjacency matrix
    Parameters
    ----------
    trimesh : numpy.ndarray
        Triads of vertices from the cortical model
    n : int
        Number of sources in model

    Returns
    -------
    adj_matrix : adjacency sparse matrix
    """
    edge_u = np.concatenate((trimesh[:, 0], trimesh[:, 0], trimesh[:, 1]))
    edge_v = np.concatenate((trimesh[:, 1], trimesh[:, 2], trimesh[:, 2]))

    # adjacency matrix - initialize with 0
    # see also: http://stackoverflow.com/questions/6667201/how-to-define-two-dimensional-array-in-python
    adj_matrix = sparse.csr_matrix((n, n))

    adj_matrix[edge_u, edge_v] = 1
    return adj_matrix
