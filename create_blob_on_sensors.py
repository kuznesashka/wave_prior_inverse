import numpy as np
import scipy.sparse.csc as csc
from create_waves_on_sensors import calculate_direction_vector
import matplotlib.pyplot as plt


def create_template_path_fixed_length(
        start_source_index: int,
        vertices: np.ndarray,
        vert_conn: csc.csc_matrix,
        vert_normals: np.ndarray,
        path_length: int = 20
):
    nearest_neighbour_list = vert_conn[start_source_index, :].nonzero()[1]
    direction_num = len(nearest_neighbour_list)

    template_path_index_array = np.zeros([direction_num, path_length], dtype=int)

    # average normal vector for all of the nearest neighbours
    normal_init = np.mean(vert_normals[nearest_neighbour_list], axis=0)[:, np.newaxis]
    normal_init = normal_init / np.linalg.norm(normal_init)

    # projection away from the average normal vector
    projector_away_from_normal_init = np.identity(3) - normal_init @ normal_init.T

    for direction_i in range(direction_num):
        selected_neighbour_ind = nearest_neighbour_list[direction_i]
        template_i = [start_source_index, selected_neighbour_ind]

        direction_vector_init = calculate_direction_vector(
            vertices=vertices,
            ind1=start_source_index,
            ind2=selected_neighbour_ind,
            projector=projector_away_from_normal_init,
        )

        neighbour_i = 2
        while neighbour_i <= path_length - 1:
            next_step_neighbour_list = vert_conn[selected_neighbour_ind, :].nonzero()[1]
            new_direction_closeness = []
            for candidate_neighbour_ind in range(len(next_step_neighbour_list)):
                direction = calculate_direction_vector(
                    vertices=vertices,
                    ind1=selected_neighbour_ind,
                    ind2=next_step_neighbour_list[candidate_neighbour_ind],
                    projector=projector_away_from_normal_init,
                )
                new_direction_closeness.append(direction @ direction_vector_init.T)

            selected_neighbour_ind = next_step_neighbour_list[np.argmax(new_direction_closeness)]
            template_i.append(selected_neighbour_ind)
            neighbour_i += 1

        template_path_index_array[direction_i, :] = template_i

    return template_path_index_array


def calculate_activation_time_series(
        total_duration: int, path_length: int = 20, plot_time_series: bool = False
):
    x = np.linspace(-1, 1, path_length * 2 + 1)
    n = len(x)
    alpha = 1
    sigma = 2

    g = np.zeros([n, n])
    for i in range(1, n):
        for j in range(1, n):
            g[i, j] = alpha * np.exp(-(x[i] ** 2 + x[j] ** 2) / 2 * sigma ** 2)

    th = np.linspace(0, 4, total_duration)
    omega = np.pi / 2
    h = 0.5 * (1 + np.cos(omega * th))

    s = np.zeros([path_length, total_duration])
    for t in range(total_duration):
        s[:, t] = np.flip(g[path_length + 1, 1: path_length + 1]) * h[t]

    if plot_time_series:
        plt.figure()
        plt.imshow(g)

        plt.figure()
        plt.plot(th, h)

        plt.figure()
        plt.plot(s.T)

    return s


def project_blob_on_sensors(
        time_series: np.ndarray,
        G: np.ndarray,
        total_duration: int,
        template_path_index_array: np.ndarray
):
    channel_num = G.shape[0]
    blob_on_sensors = np.zeros([channel_num, total_duration])
    direction_num = template_path_index_array.shape[0]

    for d in range(direction_num):
        ind = template_path_index_array[d]
        blob_on_sensors = blob_on_sensors + G[:, ind] @ time_series

    return blob_on_sensors


def create_blob_on_sensors(
        vertices: np.ndarray,
        vert_conn: csc.csc_matrix,
        vert_normals: np.ndarray,
        G: np.ndarray,
        start_source_index: int,
        total_duration: int,
        path_length: int = 20,
        plot_time_series: bool = False
):
    template_path_index_array = create_template_path_fixed_length(
        start_source_index=start_source_index,
        vertices=vertices,
        vert_conn=vert_conn,
        vert_normals=vert_normals,
        path_length=path_length
    )

    time_series = calculate_activation_time_series(
        total_duration=total_duration, path_length=path_length, plot_time_series=plot_time_series
    )

    blob_on_sensors = project_blob_on_sensors(
        time_series=time_series,
        G=G,
        total_duration=total_duration,
        template_path_index_array=template_path_index_array
    )

    return blob_on_sensors, template_path_index_array
