def create_blob_on_sensors(cortex, params, G, start_point, max_step=20):
    """Function to create static blob
        Parameters
        ----------
        cortex : numpy.ndarray
            Cortical model structure from brainstorm
        params : dict
            Duration and sampling frequency
        G : numpy.ndarray
            Forward model matrix
        start_point : int
            The wave starting vertex
        max_step : int
            Number of vertices involved into the static activation
        Returns
        -------
        sensor_blob : static blob on sensors [n_chann x T]
        path_indices : indices of vertices in path [n_dir x max_step]
        """

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    duration = params['duration']
    Fs = params['Fs']

    vertices = cortex[0][1]
    flag = 0
    p = 2
    while flag == 0:
        if cortex[0][p].shape == (G.shape[1], G.shape[1]):
            flag = 1
        p += 1
    VertConn = cortex[0][p - 1]
    VertNormals = cortex[0][p]

    # Create matrix with template paths in different directions from the starting point
    neighbour_step_1 = VertConn[start_point, :].nonzero()[1]  # nearest neighbours of the starting vertex
    num_dir = len(neighbour_step_1)  # number of propagation directions
    path_indices = np.zeros([num_dir, max_step], dtype=int)  # vertices forming the path
    for n in range(0, num_dir):
        path_indices[n, 0] = start_point
        neighbour_ind = neighbour_step_1[n]
        path_indices[n, 1] = neighbour_ind

        norm_start = np.mean(VertNormals[neighbour_step_1], axis=0)  # average normal to all of the nearest neighbours
        norm_start = norm_start[:, np.newaxis]
        norm_start = norm_start / np.linalg.norm(norm_start)
        P_norm = np.identity(3) - norm_start @ norm_start.T  # projection away from average normal

        direction_0 = vertices[neighbour_ind] - vertices[start_point]
        direction_0 = direction_0 @ P_norm.T
        direction_0 = direction_0 / np.linalg.norm(direction_0)
        d = 2
        while d <= max_step-1:
            neighbour_step_2 = VertConn[neighbour_ind, :].nonzero()[1]
            cs = np.zeros(len(neighbour_step_2))
            for p in range(0, len(neighbour_step_2)):
                direction = vertices[neighbour_step_2[p]] - vertices[neighbour_ind]
                direction = direction @ P_norm.T
                direction = direction / np.linalg.norm(direction)
                cs[p] = direction @ direction_0.T

            csmax_ind = np.argmax(cs)
            neighbour_ind = neighbour_step_2[csmax_ind]
            path_indices[n, d] = neighbour_ind
            d += 1

    # visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
    # for d in range(0, path_indices.shape[0]):
    #     ax.scatter(vertices[path_indices[d, :], 0], vertices[path_indices[d, :], 1], vertices[path_indices[d, :], 2], marker = '^')

    # Activation timeseries
    x1 = np.linspace(-1, 1, max_step*2+1)
    x2 = np.linspace(-1, 1, max_step*2+1)
    A = 1
    sigma = 2
    g = np.zeros([len(x1), len(x2)])
    for i in range(1, len(x1)):
        for j in range(1, len(x2)):
            g[i, j] = A * np.exp(-(x1[i]**2 + x2[j]**2) / 2 * sigma**2)
    # plt.figure()
    # plt.imshow(g)
    # plt.figure()
    # plt.plot(g[11, 1:11])

    ntpoints = int(Fs*duration+1)
    t = np.linspace(0, 4, ntpoints)
    omega = np.pi / 2
    h = 0.5 * (1 + np.cos(omega * t))
    # plt.figure()
    # plt.plot(t, h)

    sensor_blob = np.zeros([G.shape[0], ntpoints])
    s = np.zeros([max_step, ntpoints])
    for t in range(0, ntpoints):
        s[:, t] = np.flip(g[max_step+1, 1:max_step+1])*h[t]

    # plt.figure()
    # plt.plot(s.T)

    # activations on sensors
    for d in range(0, num_dir):
        sensor_blob = sensor_blob + G[:, path_indices[d]]@s

    return [sensor_blob, path_indices]