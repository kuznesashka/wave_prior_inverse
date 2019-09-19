import numpy as np
from tris_to_adjacency import tris_to_adjacency
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_waves_on_sensors(fwd, params, start_point, spherical=False, max_step=100):
    """Function to compute the basis waves
        Parameters
        ----------
        fwd : Forward
            Forward instance from the MNE Python
        params : dict
            Wave parameters
        start_point : int
            The wave starting vertex
        spherical : bool
            To add spherical wave or not
        max_step : int
            Maximal step for path
        Returns
        -------
        sensor_waves : waves [n_dir x n_speeds x n_chann x T]
        direction_final : direction of propagation in space [n_dir x n_speeds x 3]
        path_final : coordinates of vertices in final paths [n_dir x n_speeds x T x 3]
        """
    speeds = params['speeds']
    duration = params['duration']
    Fs = params['Fs']

    if start_point <= fwd['src'][0]['nuse']:
        hemi_idx = 0
        G = fwd['sol']['data'][:, 0:fwd['src'][0]['nuse']]
    else:
        hemi_idx = 1
        G = fwd['sol']['data'][:, fwd['src'][0]['nuse']:-1]
        start_point = start_point - fwd['src'][0]['nuse'] - 1

    vert_idx = fwd['src'][hemi_idx]['vertno']
    vertices = fwd['src'][hemi_idx]['rr'][vert_idx, :]

    trimesh_global = fwd['src'][hemi_idx]['use_tris']
    trimesh = np.zeros_like(trimesh_global)
    for key, val in zip(vert_idx, np.arange(0, vert_idx.shape[0])):
        trimesh[trimesh_global == key] = val

    VertConn = tris_to_adjacency(trimesh, vertices.shape[0])
    VertNormals = fwd['src'][hemi_idx]['nn'][vert_idx, :]

    # Create matrix with template paths in different directions from the starting point
    neighbour_step_1 = VertConn[start_point, :].nonzero()[1]  # nearest neighbours of the starting vertex
    num_dir = len(neighbour_step_1)  # number of propagation directions
    path_indices = np.zeros([num_dir, max_step], dtype=int)  # vertices forming the path

    ntpoints = int(Fs * duration) + 1  # number of time points to generate
    path_final = np.zeros([num_dir, len(speeds), ntpoints, 3])
    path_sphere_final = np.zeros([num_dir, len(speeds), ntpoints, 3])
    forward_model = np.zeros([num_dir, len(speeds), ntpoints, G.shape[0]])
    direction_curved = np.zeros([num_dir, len(speeds), 3])
    direction_pca = np.zeros([num_dir, len(speeds), 3])
    direction_on_sphere = np.zeros([num_dir, len(speeds), 3])
    projected_path = np.zeros([num_dir, len(speeds), ntpoints, 2])
    tstep = 1 / Fs

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

        # compute distance to the following point (for all points in a path)
        first = vertices[path_indices[n, :-1]]
        next = vertices[path_indices[n, 1:]]
        dist = np.sqrt(np.sum((next - first) ** 2, axis=1))

        for s in range(0, len(speeds)):
            l = speeds[s] * tstep
            path_final[n, s, 0, :] = vertices[start_point]
            path_sphere_final[n, s, 0, :] = VertNormals[start_point]
            forward_model[n, s, 0, :] = G[:, start_point]
            res = 0
            v1 = 0
            v2 = 1
            for t in range(1, ntpoints):
                if l < res:
                    alpha = 1 - l / res
                    path_final[n, s, t, :] = alpha * path_final[n, s, (t - 1), :] + (1 - alpha) * vertices[
                        path_indices[n, v2]]
                    path_sphere_final[n, s, t, :] = alpha * path_sphere_final[n, s, (t - 1), :] + (1 - alpha) * VertNormals[
                        path_indices[n, v2]]
                    forward_model[n, s, t, :] = alpha * forward_model[n, s, (t - 1), :] + (1 - alpha) * G[:,
                                                                                                        path_indices[
                                                                                                            n, v2]]
                    res = res - l
                elif l > res:
                    if res == 0:
                        if l < dist[(v2 - 1)]:
                            alpha = 1 - l / dist[(v2 - 1)]
                            path_final[n, s, t, :] = alpha * vertices[path_indices[n, v1]] + (1 - alpha) * vertices[
                                path_indices[n, v2]]
                            path_sphere_final[n, s, t, :] = alpha * VertNormals[path_indices[n, v1]] + (1 - alpha) * VertNormals[
                                path_indices[n, v2]]
                            forward_model[n, s, t, :] = alpha * G[:, path_indices[n, v1]] + (1 - alpha) * G[:,
                                                                                                          path_indices[
                                                                                                              n, v2]]
                            res = dist[(v2 - 1)] - l
                        elif l == dist[(v2 - 1)]:
                            path_final[n, s, t, :] = vertices[path_indices[n, v2]]
                            path_sphere_final[n, s, t, :] = VertNormals[path_indices[n, v2]]
                            forward_model[n, s, t, :] = G[:, path_indices[n, v2]]
                            v1 += 1
                            v2 += 1
                        else:
                            l2 = l - dist[(v2 - 1)]
                            v1 += 1
                            v2 += 1
                            while l2 > dist[(v2 - 1)]:
                                l2 = l2 - dist[(v2 - 1)]
                                v1 += 1
                                v2 += 1
                            alpha = 1 - l2 / dist[(v2 - 1)]
                            path_final[n, s, t, :] = alpha * vertices[path_indices[n, v1]] + (1 - alpha) * vertices[
                                path_indices[n, v2]]
                            path_sphere_final[n, s, t, :] = alpha * VertNormals[path_indices[n, v1]] + (1 - alpha) * VertNormals[
                                path_indices[n, v2]]
                            forward_model[n, s, t, :] = alpha * G[:, path_indices[n, v1]] + (1 - alpha) * G[:,
                                                                                                          path_indices[
                                                                                                              n, v2]]
                            res = dist[(v2 - 1)] - l2
                    else:
                        l2 = l - res
                        v1 += 1
                        v2 += 1
                        while l2 > dist[(v2 - 1)]:
                            l2 = l2 - dist[(v2 - 1)]
                            v1 += 1
                            v2 += 1
                        alpha = 1 - l2 / dist[(v2 - 1)]
                        path_final[n, s, t, :] = alpha * vertices[path_indices[n, v1]] + (1 - alpha) * vertices[
                            path_indices[n, v2]]
                        path_sphere_final[n, s, t, :] = alpha * VertNormals[path_indices[n, v1]] + (1 - alpha) * VertNormals[
                            path_indices[n, v2]]
                        forward_model[n, s, t, :] = alpha * G[:, path_indices[n, v1]] + (1 - alpha) * G[:, path_indices[
                                                                                                               n, v2]]
                        res = dist[(v2 - 1)] - l2
                else:
                    path_final[n, s, t, :] = vertices[path_indices[n, v2]]
                    path_sphere_final[n, s, t, :] = VertNormals[path_indices[n, v2]]
                    forward_model[n, s, t, :] = G[:, path_indices[n, v2]]
                    v1 += 1
                    v2 += 1

            direction_on_sphere[n, s, :] = path_sphere_final[n, s, -1, :] - path_sphere_final[n, s, 0, :]
            direction_curved[n, s, :] = path_final[n, s, -1, :] - path_final[n, s, 0, :]

            [u, ll, v] = np.linalg.svd(path_final[n, s, :, :])
            direction_pca[n, s, :] = u[:, 1].T@path_final[n, s, :, :]

            [u, ll, v] = np.linalg.svd(path_sphere_final[n, s, :, :])
            projected_path[n, s, :, :] = path_final[n, s, :, :]@v[:, 0:2]

    # visualization for comparison of curved brain and spherical model
    # d = 0
    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(vertices[0:-1:100, 0], vertices[0:-1:100, 1], vertices[0:-1:100, 2])
    # for i in range(0, 10):
    #     ax.scatter(vertices[path_indices[d, i], 0], vertices[path_indices[d, i], 1], vertices[path_indices[d, i], 2], s=100, marker='^')
    # ax = fig.add_subplot(122, projection='3d')
    # ax.scatter(VertNormals[0:-1:100, 0], VertNormals[0:-1:100, 1], VertNormals[0:-1:100, 2])
    # for i in range(0, 10):
    #     ax.scatter(VertNormals[path_indices[d, i], 0], VertNormals[path_indices[d, i], 1], VertNormals[path_indices[d, i], 2],
    #                s=100, marker='^')

    # source time series
    t = np.arange(0, ntpoints * 2 / 100, 1 / 100)
    k = np.arange(0, ntpoints * 1 / 100, 1 / 100)
    p = np.tile(t, (len(k), 1))-np.tile(k.T, (len(t), 1)).T
    wave = np.sin(10 * np.pi * p) * np.exp(-10 * (2 * p + 0.2) ** 2)
    for i in range(0, ntpoints):
        wave[i, :i] = np.zeros(i)

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

    return [sensor_waves, direction_curved, direction_pca, direction_on_sphere, projected_path]
