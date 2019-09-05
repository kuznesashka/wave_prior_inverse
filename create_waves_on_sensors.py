import numpy as np
import matplotlib.pyplot as plt


def create_waves_on_sensors(cortex, params, G, start_point, spherical=0, max_step=100):
    """Function to compute the basis waves
        Parameters
        ----------
        cortex : numpy.ndarray
            Cortical model structure from brainstorm
        params : dict
            Wave parameters
        G : numpy.ndarray
            Forward model matrix
        start_point : int
            The wave starting vertex
        spherical : bool
            To add spherical wave or not
        max_step : int
            Maximal step for path
        Returns
        -------
        sensor_waves : waves [n_dir x n_speeds x n_chann x T]
        path_indices : indices of vertices in path [n_dir x max_step]
        path_final : coordinates of vertices in final paths [n_dir x n_speeds x T x 3]
        """

    speeds = params['speeds']
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
        while d <= max_step - 1:
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

    # for all directions compute distance to the following point
    first = vertices[path_indices[:, :-1]]
    next = vertices[path_indices[:, 1:]]
    dist = np.sqrt(np.sum((next - first) ** 2, axis=2))

    ntpoints = int(Fs * duration) + 1  # number of time points to generate
    path_final = np.zeros([num_dir, len(speeds), ntpoints, 3])
    forward_model = np.zeros([num_dir, len(speeds), ntpoints, G.shape[0]])
    tstep = 1 / Fs

    for s in range(0, len(speeds)):
        l = speeds[s] * tstep
        for d in range(0, num_dir):
            path_final[d, s, 0, :] = vertices[start_point]
            forward_model[d, s, 0, :] = G[:, start_point]
            res = 0
            v1 = 0
            v2 = 1
            for t in range(1, ntpoints):
                if l < res:
                    alpha = 1 - l / res
                    path_final[d, s, t, :] = alpha * path_final[d, s, (t - 1), :] + (1 - alpha) * vertices[
                        path_indices[d, v2]]
                    forward_model[d, s, t, :] = alpha * forward_model[d, s, (t - 1), :] + (1 - alpha) * G[:,
                                                                                                        path_indices[
                                                                                                            d, v2]]
                    res = res - l
                elif l > res:
                    if res == 0:
                        if l < dist[d, (v2 - 1)]:
                            alpha = 1 - l / dist[d, (v2 - 1)]
                            path_final[d, s, t, :] = alpha * vertices[path_indices[d, v1]] + (1 - alpha) * vertices[
                                path_indices[d, v2]]
                            forward_model[d, s, t, :] = alpha * G[:, path_indices[d, v1]] + (1 - alpha) * G[:,
                                                                                                          path_indices[
                                                                                                              d, v2]]
                            res = dist[d, (v2 - 1)] - l
                        elif l == dist[d, (v2 - 1)]:
                            path_final[d, s, t, :] = vertices[path_indices[d, v2]]
                            forward_model[d, s, t, :] = G[:, path_indices[d, v2]]
                            v1 += 1
                            v2 += 1
                        else:
                            l2 = l - dist[d, (v2 - 1)]
                            v1 += 1
                            v2 += 1
                            while l2 > dist[d, (v2 - 1)]:
                                l2 = l2 - dist[d, (v2 - 1)]
                                v1 += 1
                                v2 += 1
                            alpha = 1 - l2 / dist[d, (v2 - 1)]
                            path_final[d, s, t, :] = alpha * vertices[path_indices[d, v1]] + (1 - alpha) * vertices[
                                path_indices[d, v2]]
                            forward_model[d, s, t, :] = alpha * G[:, path_indices[d, v1]] + (1 - alpha) * G[:,
                                                                                                          path_indices[
                                                                                                              d, v2]]
                            res = dist[d, (v2 - 1)] - l2
                    else:
                        l2 = l - res
                        v1 += 1
                        v2 += 1
                        while l2 > dist[d, (v2 - 1)]:
                            l2 = l2 - dist[d, (v2 - 1)]
                            v1 += 1
                            v2 += 1
                        alpha = 1 - l2 / dist[d, (v2 - 1)]
                        path_final[d, s, t, :] = alpha * vertices[path_indices[d, v1]] + (1 - alpha) * vertices[
                            path_indices[d, v2]]
                        forward_model[d, s, t, :] = alpha * G[:, path_indices[d, v1]] + (1 - alpha) * G[:, path_indices[
                                                                                                               d, v2]]
                        res = dist[d, (v2 - 1)] - l2
                else:
                    path_final[d, s, t, :] = vertices[path_indices[d, v2]]
                    forward_model[d, s, t, :] = G[:, path_indices[d, v2]]
                    v1 += 1
                    v2 += 1

    # source timeseries
    # TODO: replace loop
    t = np.arange(0, ntpoints * 2 / 100, 1 / 100)
    k = np.arange(0, ntpoints * 1 / 100, 1 / 100)
    wave = np.zeros([ntpoints, len(t)])
    for i in range(0, ntpoints):
        wave[i] = np.sin(10 * np.pi * (t - k[i])) * np.exp(-10 * (2 * (t - k[i]) + 0.2) ** 2)
        wave[i, :i] = np.zeros(i)

    # l = np.tile(t, (len(k), 1))-np.tile(k.T, (len(t), 1)).T
    # wave = np.sin(10 * np.pi * l) * np.exp(-10 * (2 * l + 0.2) ** 2)

    # plt.figure()
    # plt.plot(t, wave.T, 'k')
    # plt.plot(t, wave[0], 'r', label='First source', lw=3)
    # plt.xlabel('Time, ms')
    # plt.ylabel('Amplitude, a.u.')

    T = wave.shape[1]
    if spherical == 1:
        sensor_waves = np.zeros([num_dir + 1, len(speeds), G.shape[0], T])
    else:
        sensor_waves = np.zeros([num_dir, len(speeds), G.shape[0], T])
    for s in range(0, len(speeds)):
        for i in range(0, num_dir):
            fm_s = np.zeros([G.shape[0], ntpoints])
            for k in range(0, ntpoints):
                fm_s[:, k] = forward_model[i, s, k, :]
            A = fm_s @ wave
            sensor_waves[i, s, :, :] = A
    if spherical == 1:
        for s in range(0, len(speeds)):
            for i in range(0, num_dir):
                sensor_waves[num_dir, s, :, :] = sensor_waves[num_dir, s, :, :] + sensor_waves[i, s, :, :]
            sensor_waves[num_dir, s, :, :] = sensor_waves[num_dir, s, :, :] / num_dir

    return [sensor_waves, path_indices, path_final]
