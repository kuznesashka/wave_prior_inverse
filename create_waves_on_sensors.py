import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_waves_on_sensors(
    cortex: np.ndarray,
    cortex_smooth: np.ndarray,
    params: Dict[str, Any],
    G: np.ndarray,
    start_point: int,
    spherical: bool = False,
    max_step: int = 100
):
    """Function to compute basis waves.
    Parameters
    ----------
    cortex : np.ndarray
        Cortical model structure from brainstorm
    cortex_smooth : np.ndarray
    params : Dict[str, Any]
        Wave parameters
    G : np.ndarray
        Forward model matrix
    start_point : int
        The wave starting vertex
    spherical : bool
        To add spherical wave or not
    max_step : int
        Maximal step for path

    Returns
    -------
    sensor_waves :
        waves [n_dir x n_speeds x n_chann x T]
    direction_final :
        direction of propagation in space [n_dir x n_speeds x 3]
    path_final :
        coordinates of vertices in final paths [n_dir x n_speeds x T x 3]
    """

    # wave parameters
    speed_list = params["speeds"]
    duration = params["duration"]
    fs = params["Fs"]

    # vertices and connections between them in cortical model
    vertices = cortex["Vertices"][0]
    vertices_smooth = cortex_smooth["Vertices"][0]
    assert vertices.shape == vertices_smooth.shape, "smooth cortical model does not correspond to initial"

    vert_conn = cortex["VertConn"][0]
    assert vert_conn.shape == (G.shape[1], G.shape[1]), "cortical model does not correspond to forward model"

    vert_normals = cortex["VertNormals"][0]
    assert vert_normals.shape == vertices.shape, "cortical model is broken"

    # Create matrix with template paths in different directions from the starting point
    neighbour_step_1 = vert_conn[start_point, :].nonzero()[1]  # nearest neighbours of the starting vertex
    num_dir = len(neighbour_step_1)  # number of propagation directions
    path_indices = np.zeros([num_dir, max_step], dtype=int)  # vertices forming the path

    speed_num = len(speed_list)
    ntpoints = int(fs * duration) + 1  # number of time points to generate
    path_final = np.zeros([num_dir, speed_num, ntpoints, 3])
    path_final_smooth = np.zeros([num_dir, speed_num, ntpoints, 3])
    forward_model = np.zeros([num_dir, speed_num, ntpoints, G.shape[0]])
    direction_final = np.zeros([num_dir, speed_num, 3])
    direction_final_smooth = np.zeros([num_dir, speed_num, 3])
    direction_pca = np.zeros([num_dir, speed_num, 3])
    tstep = 1 / fs

    for n in range(num_dir):

        # templates of paths
        path_indices[n, 0] = start_point
        neighbour_ind = neighbour_step_1[n]
        path_indices[n, 1] = neighbour_ind

        norm_start = np.mean(
            vert_normals[neighbour_step_1], axis=0
        )  # average normal to all of the nearest neighbours
        norm_start = norm_start[:, np.newaxis]
        norm_start = norm_start / np.linalg.norm(norm_start)

        p_norm = (
            np.identity(3) - norm_start @ norm_start.T
        )  # projection away from average normal

        direction_0 = vertices[neighbour_ind] - vertices[start_point]
        direction_0 = direction_0 @ p_norm.T
        direction_0 = direction_0 / np.linalg.norm(direction_0)

        d = 2
        while d <= max_step - 1:
            neighbour_step_2 = vert_conn[neighbour_ind, :].nonzero()[1]
            cs = np.zeros(len(neighbour_step_2))
            for p in range(len(neighbour_step_2)):
                direction = vertices[neighbour_step_2[p]] - vertices[neighbour_ind]
                direction = direction @ p_norm.T
                direction = direction / np.linalg.norm(direction)
                cs[p] = direction @ direction_0.T

            csmax_ind = np.argmax(cs)
            neighbour_ind = neighbour_step_2[csmax_ind]
            path_indices[n, d] = neighbour_ind
            d += 1

        # compute distance to the following point
        first = vertices[path_indices[n, :-1]]
        next = vertices[path_indices[n, 1:]]
        dist = np.sqrt(np.sum((next - first) ** 2, axis=1))

        # final paths considering the speed values
        for s in range(0, speed_num):
            l = speed_list[s] * tstep
            path_final[n, s, 0, :] = vertices[start_point]
            path_final_smooth[n, s, 0, :] = vertices_smooth[start_point]
            forward_model[n, s, 0, :] = G[:, start_point]
            res = 0
            v1 = 0
            v2 = 1
            for t in range(1, ntpoints):
                if l < res:
                    alpha = 1 - l / res
                    path_final[n, s, t, :] = (
                        alpha * path_final[n, s, (t - 1), :]
                        + (1 - alpha) * vertices[path_indices[n, v2]]
                    )
                    path_final_smooth[n, s, t, :] = (
                        alpha * path_final_smooth[n, s, (t - 1), :]
                        + (1 - alpha) * vertices_smooth[path_indices[n, v2]]
                    )
                    forward_model[n, s, t, :] = (
                        alpha * forward_model[n, s, (t - 1), :]
                        + (1 - alpha) * G[:, path_indices[n, v2]]
                    )
                    res = res - l
                elif l > res:
                    if res == 0:
                        if l < dist[(v2 - 1)]:
                            alpha = 1 - l / dist[(v2 - 1)]
                            path_final[n, s, t, :] = (
                                alpha * vertices[path_indices[n, v1]]
                                + (1 - alpha) * vertices[path_indices[n, v2]]
                            )
                            path_final_smooth[n, s, t, :] = (
                                alpha * vertices_smooth[path_indices[n, v1]]
                                + (1 - alpha) * vertices_smooth[path_indices[n, v2]]
                            )
                            forward_model[n, s, t, :] = (
                                alpha * G[:, path_indices[n, v1]]
                                + (1 - alpha) * G[:, path_indices[n, v2]]
                            )
                            res = dist[(v2 - 1)] - l
                        elif l == dist[(v2 - 1)]:
                            path_final[n, s, t, :] = vertices[path_indices[n, v2]]
                            path_final_smooth[n, s, t, :] = vertices_smooth[
                                path_indices[n, v2]
                            ]
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
                            path_final[n, s, t, :] = (
                                alpha * vertices[path_indices[n, v1]]
                                + (1 - alpha) * vertices[path_indices[n, v2]]
                            )
                            path_final_smooth[n, s, t, :] = (
                                alpha * vertices_smooth[path_indices[n, v1]]
                                + (1 - alpha) * vertices_smooth[path_indices[n, v2]]
                            )
                            forward_model[n, s, t, :] = (
                                alpha * G[:, path_indices[n, v1]]
                                + (1 - alpha) * G[:, path_indices[n, v2]]
                            )
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
                        path_final[n, s, t, :] = (
                            alpha * vertices[path_indices[n, v1]]
                            + (1 - alpha) * vertices[path_indices[n, v2]]
                        )
                        path_final_smooth[n, s, t, :] = (
                            alpha * vertices_smooth[path_indices[n, v1]]
                            + (1 - alpha) * vertices_smooth[path_indices[n, v2]]
                        )
                        forward_model[n, s, t, :] = (
                            alpha * G[:, path_indices[n, v1]]
                            + (1 - alpha) * G[:, path_indices[n, v2]]
                        )
                        res = dist[(v2 - 1)] - l2
                else:
                    path_final[n, s, t, :] = vertices[path_indices[n, v2]]
                    path_final_smooth[n, s, t, :] = vertices_smooth[path_indices[n, v2]]
                    forward_model[n, s, t, :] = G[:, path_indices[n, v2]]
                    v1 += 1
                    v2 += 1

            direction_final[n, s, :] = path_final[n, s, -1, :] - path_final[n, s, 0, :]
            direction_final_smooth[n, s, :] = (
                path_final_smooth[n, s, -1, :] - path_final_smooth[n, s, 0, :]
            )

            [u, _, _] = np.linalg.svd(path_final[n, s, :, :])
            direction_pca[n, s, :] = u[:, 1].T @ path_final[n, s, :, :]

    # visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(vertices[0:-1:100, 0], vertices[0:-1:100, 1], vertices[0:-1:100, 2])
    # for d in range(0, path_final.shape[0]):
    #     ax.scatter(path_final[d, s, :, 0], path_final[d, s, :, 1], path_final[d, s, :, 2], marker = '^')

    # source time series
    t = np.arange(0, ntpoints * 2 / 100, 1 / 100)
    k = np.arange(0, ntpoints * 1 / 100, 1 / 100)
    p = np.tile(t, (len(k), 1)) - np.tile(k.T, (len(t), 1)).T
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
        sensor_waves = np.zeros([num_dir + 1, speed_num, G.shape[0], T])
    else:
        sensor_waves = np.zeros([num_dir, speed_num, G.shape[0], T])
    for s in range(0, speed_num):
        for i in range(0, num_dir):
            fm_s = np.zeros([G.shape[0], ntpoints])
            for k in range(0, ntpoints):
                fm_s[:, k] = forward_model[i, s, k, :]
            A = fm_s @ wave
            sensor_waves[i, s, :, :] = A
    if spherical == 1:
        for s in range(0, speed_num):
            for i in range(0, num_dir):
                sensor_waves[num_dir, s, :, :] = (
                    sensor_waves[num_dir, s, :, :] + sensor_waves[i, s, :, :]
                )
            sensor_waves[num_dir, s, :, :] = sensor_waves[num_dir, s, :, :] / num_dir

    return [
        sensor_waves,
        path_final,
        path_final_smooth,
        direction_final,
        direction_final_smooth,
        direction_pca,
    ]
