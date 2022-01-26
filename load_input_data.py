import scipy.io
import numpy as np


def load_input_data(data_dir: str, channel_type: str):
    """Import cortical model and forward operators precalculated in Brainstorm.

    Parameters
    ----------
    data_dir : str
        Data directory with precalculated in Brainstorm:
        - G.mat (forward operator, low source number)
        - cortex.mat (cortical model, low source number)
        - cortex_smooth.mat (smooth_cortex, low source number)
        - G_dense.mat (forward_operator, high source number)
        - cortes_dense.mat (cortical model, high source number)
        - cortex_smooth_dense.mat (smooth cortex, high source number)
    channel_type : str
        MEG channels to use: 'grad' or 'mag'.

    Returns
    -------
    G_dense : np.ndarray
        Forward operator for dense cortical model [n_channels x n_sources_dense].
        Source orientation is fixed.
    G : np.ndarray
        Forward operator for sparce cortical model [n_channels x n_sources_sparse].
        Source orientation is fixed.
    cortex_dense : np.ndarray
        Dense cortical model.
    cortex : np.ndarray
        Sparse cortical model.
    cortex_smooth_dense : np.ndarray
        Dense smooth cortical model.
    cortex_smooth : np.ndarray
        Sparce smooth cortical model.
    vertices_dense : np.ndarray
        Vertex coordinates for dense model [n_sources_dense x 3].
    vertices_smooth_dense
    vertices : np.ndarray
        Vertex coordinates for sparse model [n_sources_sparse x 3].
    vertices_smooth
    vert_conn
    vert_conn_dense
    vert_normals
    vert_normals_dense
    """
    # Dense cortical grid
    G_dense = scipy.io.loadmat(data_dir + "/G_dense.mat")["G"]
    cortex_dense = scipy.io.loadmat(data_dir + "/cortex_dense.mat")["cortex"][0]
    cortex_smooth_dense = scipy.io.loadmat(data_dir + "/cortex_smooth_dense.mat")["cortex"][0]

    # Sparse cortical grid
    G = scipy.io.loadmat(data_dir + "/G.mat")["G"]
    cortex = scipy.io.loadmat(data_dir + "/cortex.mat")["cortex"][0]
    cortex_smooth = scipy.io.loadmat(data_dir + "/cortex_smooth.mat")["cortex"][0]

    # Select channels according to the channel_type
    if channel_type == "mag":
        mag_indices = np.arange(2, 306, 3)
        G_dense = G_dense[mag_indices]
        G = G[mag_indices]
    elif channel_type == "grad":
        grad_indices = np.setdiff1d(range(0, 306), np.arange(2, 306, 3))
        G_dense = G_dense[grad_indices]
        G = G[grad_indices]
    else:
        print("Wrong channel name")

    vertices = cortex["Vertices"][0]
    vertices_dense = cortex_dense["Vertices"][0]
    vertices_smooth = cortex_smooth["Vertices"][0]
    assert vertices.shape == vertices_smooth.shape, "smooth cortical model does not correspond to initial model"
    vertices_smooth_dense = cortex_smooth_dense["Vertices"][0]
    assert vertices_dense.shape == vertices_smooth_dense.shape, \
        "dense smooth cortical model does not correspond to initial model"

    vert_conn = cortex["VertConn"][0]
    assert vert_conn.shape == (G.shape[1], G.shape[1]), "cortical model does not correspond to forward model"
    vert_conn_dense = cortex_dense["VertConn"][0]
    assert vert_conn_dense.shape == (G_dense.shape[1], G_dense.shape[1]), \
        "dense cortical model does not correspond to dense forward model"

    vert_normals = cortex["VertNormals"][0]
    assert vert_normals.shape == vertices.shape, "cortical model is broken"
    vert_normals_dense = cortex_dense["VertNormals"][0]
    assert vert_normals_dense.shape == vertices_dense.shape, "dense cortical model is broken"

    return (
        G_dense,
        G,
        cortex_dense,
        cortex,
        cortex_smooth_dense,
        cortex_smooth,
        vertices,
        vertices_dense,
        vertices_smooth,
        vertices_smooth_dense,
        vert_conn,
        vert_conn_dense,
        vert_normals,
        vert_normals_dense
    )
