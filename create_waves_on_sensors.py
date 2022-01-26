import scipy.sparse.csc as csc
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt


def calculate_direction_vector(
    vertices: np.ndarray, ind1: int, ind2: int, projector: np.ndarray
) -> np.ndarray:
    """Calculate propagation direction coordinates vector from source with ind1 to source with ind2
    and projected to the plane defined by starting source nearest neighbours.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates from cortical model.
    ind1 : int
        First source index.
    ind2 : int
        Second source index.
    projector : np.ndarray
        Projector to the plane defined by starting source nearest neighbours.

    Returns
    -------
    direction_vector : np.ndarray
        Vector defining propagation direction coordinates [1 x 3]
    """
    direction_vector = vertices[ind2] - vertices[ind1]
    direction_vector = direction_vector @ projector.T
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    return direction_vector


def create_template_paths(
    wave_generation_params: Dict[str, Any],
    start_source_index: int,
    vertices: np.ndarray,
    vert_conn: csc.csc_matrix,
    vert_normals: np.ndarray,
):
    """Calculate template cortical paths for waves.
    Consider only propagation directions, but not actual propagation speed.
    For each direction calculates the path with the length sufficient to wave with the maximal speed.
    Calculates the distance to the next source on the path as well.

    Parameters
    ----------
    wave_generation_params : Dict[str, Any]
        Wave parameters.
    start_source_index : int
        Index of the wave starting source.
    vertices : np.ndarray
        Vertex coordinates from cortical model [sources_num x 3].
    vert_conn : csc.csc_matrix
        Sparse matrix with vertex connections [sources_num x sources_num].
    vert_normals : np.ndarray
        Normal vector coordinates for each source [sources_num x 3].

    Returns
    -------
    template_path_index_array : Dict[int, List[int]]
        Cortical path templates for each propagation direction.
    distance_to_next_source : Dict[int, List[float]]
        Distance to the next source on the path for each propagation direction.
    """
    speed_list = wave_generation_params["speeds"]
    max_distance = max(speed_list) * wave_generation_params["duration"]

    # nearest neighbours of the starting vertex
    nearest_neighbour_list = vert_conn[start_source_index, :].nonzero()[1]
    propagation_direction_num = len(nearest_neighbour_list)

    # create empty dictionary for template paths propagating in different directions from the starting point
    template_path_index_array = dict()
    distance_to_next_source = dict()

    # average normal vector for all of the nearest neighbours
    normal_init = np.mean(vert_normals[nearest_neighbour_list], axis=0)[:, np.newaxis]
    normal_init = normal_init / np.linalg.norm(normal_init)

    # projection away from the average normal vector
    projector_away_from_normal_init = np.identity(3) - normal_init @ normal_init.T

    for direction_i in range(propagation_direction_num):
        selected_neighbour_ind = nearest_neighbour_list[direction_i]
        template_i = [start_source_index, selected_neighbour_ind]
        total_distance_passed = np.linalg.norm(
            vertices[selected_neighbour_ind] - vertices[start_source_index]
        )
        distance_i = [total_distance_passed]

        direction_vector_init = calculate_direction_vector(
            vertices=vertices,
            ind1=start_source_index,
            ind2=selected_neighbour_ind,
            projector=projector_away_from_normal_init,
        )

        neighbour_i = 2
        while total_distance_passed <= max_distance:
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

            previous_source_ind = selected_neighbour_ind
            selected_neighbour_ind = next_step_neighbour_list[np.argmax(new_direction_closeness)]
            template_i.append(selected_neighbour_ind)
            dist_to_new_neighbour = np.linalg.norm(
                vertices[selected_neighbour_ind] - vertices[previous_source_ind]
            )
            distance_i.append(dist_to_new_neighbour)
            total_distance_passed += dist_to_new_neighbour
            neighbour_i += 1

        template_path_index_array[direction_i] = template_i
        distance_to_next_source[direction_i] = distance_i

    return template_path_index_array, distance_to_next_source


def plot_template_paths(
    vertices: np.ndarray, template_path_index_array: Dict[int, List[int]]
):
    """TBD: Visualize the constructed cortical paths.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates from cortical model [sources_num x 3].
    template_path_index_array : Dict[int, List[int]]
        Cortical path templates for each propagation direction.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[0:-1:100, 0], vertices[0:-1:100, 1], vertices[0:-1:100, 2])
    for direction_i in template_path_index_array:
        vertex_index_list = template_path_index_array[direction_i]
        ax.scatter(
            vertices[vertex_index_list, 0],
            vertices[vertex_index_list, 1],
            vertices[vertex_index_list, 2],
            marker="^",
        )


def one_step_lower_than_residual(
    v2: int,
    G: np.ndarray,
    vertices: np.ndarray,
    vertices_smooth: np.ndarray,
    distance_residual: float,
    one_step_distance: float,
    coord_previous: np.ndarray,
    coord_smooth_previous: np.ndarray,
    forward_model_previous: np.ndarray,
    template_path_index_direction_i: List[int],
):
    """Update wave path parameters after one step in case
    that one step distance is lower than the residual distance to the next source.

    Parameters
    ----------
    v2 : int
        Order index of the next source on the wave path.
    G : np.ndarray
        Forward model matrix [channels_num x sources_num].
    vertices : np.ndarray
        Vertex coordinates from cortical model [sources_num x 3].
    vertices_smooth
        Vertex coordinates from smooth cortical model [sources_num x 3].
    distance_residual : float
        Distance left to the next source.
    one_step_distance : float
        Distance the wave travels in one step.
    coord_previous : np.ndarray
        Coordinates of the previous point on the path.
    coord_smooth_previous : np.ndarray
        Coordinates of the previous point on the path (smooth cortical model).
    forward_model_previous : np.ndarray
        Updated forward model vector for the prevoius point on the path.
    template_path_index_direction_i : List[int]
        Indices of sources forming the path to the i-th direction.

    Returns
    -------
    distance_residual : float
    coord_next : np.ndarray
    coord_smooth_next : np.ndarray
    forward_model_next : np.ndarray
    """
    alpha = 1 - one_step_distance / distance_residual
    v2_source_ind = template_path_index_direction_i[v2]

    coord_next = alpha * coord_previous + (1 - alpha) * vertices[v2_source_ind]
    coord_smooth_next = alpha * coord_smooth_previous + (1 - alpha) * vertices_smooth[v2_source_ind]
    forward_model_next = alpha * forward_model_previous + (1 - alpha) * G[:, v2_source_ind]
    distance_residual = distance_residual - one_step_distance

    return distance_residual, coord_next, coord_smooth_next, forward_model_next


def one_step_higher_than_distance_to_next_source(
    v1: int,
    v2: int,
    one_step_distance: float,
    distance_to_go: float,
    distance_to_next_source_direction_i: List[float],
):
    """Update wave path parameters after one step in case
    that one step distance is higher than the residual distance to the next source.

    Parameters
    ----------
    v1
    v2
    one_step_distance
    distance_to_go
    distance_to_next_source_direction_i

    Returns
    -------

    """
    one_step_residual = one_step_distance - distance_to_go
    v1 += 1
    v2 += 1
    distance_to_go = distance_to_next_source_direction_i[(v2 - 1)]
    while one_step_residual > distance_to_go:
        one_step_residual = one_step_residual - distance_to_go
        v1 += 1
        v2 += 1
        distance_to_go = distance_to_next_source_direction_i[(v2 - 1)]
    return v1, v2, one_step_residual, distance_to_go


def one_step_higher_than_residual(
    v1: int,
    v2: int,
    G: np.ndarray,
    vertices: np.ndarray,
    vertices_smooth: np.ndarray,
    distance_residual: float,
    one_step_distance: float,
    template_path_index_direction_i: List[int],
    distance_to_next_source_direction_i: List[float],
):
    """

    Parameters
    ----------
    v1 : int
        Order index of the first vertex on the edge the wave travels at the moment.
    v2 : int
        Order index of the second vertex on the edge the wave travels at the moment.
    G : np.ndarray
        Forward model matrix.
    vertices : np.ndarray
        Vertex coordinates from cortical model.
    vertices_smooth : np.ndarray
        Vertices coordinates in the smooth cortical model.
    distance_residual : float
        Distance left to reach v2.
    one_step_distance : float
        Distance wave completes in one step.
    template_path_index_direction_i : List[int]
    distance_to_next_source_direction_i

    Returns
    -------

    """
    distance_to_go = distance_to_next_source_direction_i[(v2 - 1)]
    if distance_residual == 0 and one_step_distance == distance_to_go:
        v2_source_ind = template_path_index_direction_i[v2]
        coord_next = vertices[v2_source_ind]
        coord_smooth_next = vertices_smooth[v2_source_ind]
        forward_model_next = G[:, v2_source_ind]
        v1 += 1
        v2 += 1
    else:
        if distance_residual == 0 and one_step_distance > distance_to_go:
            (
                v1,
                v2,
                one_step_residual,
                distance_to_go,
            ) = one_step_higher_than_distance_to_next_source(
                v1=v1,
                v2=v2,
                one_step_distance=one_step_distance,
                distance_to_go=distance_to_go,
                distance_to_next_source_direction_i=distance_to_next_source_direction_i,
            )
            alpha = 1 - one_step_residual / distance_to_go
            distance_residual = distance_to_go - one_step_residual
        elif distance_residual == 0 and one_step_distance < distance_to_go:
            alpha = 1 - one_step_distance / distance_to_go
            distance_residual = distance_to_go - one_step_distance
        else:
            (
                v1,
                v2,
                one_step_residual,
                distance_to_go,
            ) = one_step_higher_than_distance_to_next_source(
                v1=v1,
                v2=v2,
                one_step_distance=one_step_distance,
                distance_to_go=distance_residual,
                distance_to_next_source_direction_i=distance_to_next_source_direction_i,
            )
            alpha = 1 - one_step_residual / distance_to_go
            distance_residual = distance_to_go - one_step_residual

        v1_source_ind = template_path_index_direction_i[v1]
        v2_source_ind = template_path_index_direction_i[v2]
        coord_next = alpha * vertices[v1_source_ind] + (1 - alpha) * vertices[v2_source_ind]
        coord_smooth_next = (
            alpha * vertices_smooth[v1_source_ind]
            + (1 - alpha) * vertices_smooth[v2_source_ind]
        )
        forward_model_next = alpha * G[:, v1_source_ind] + (1 - alpha) * G[:, v2_source_ind]

    return (
        v1,
        v2,
        distance_residual,
        coord_next,
        coord_smooth_next,
        forward_model_next,
    )


def one_step_equals_to_residual(
    v1: int,
    v2: int,
    G: np.ndarray,
    vertices: np.ndarray,
    vertices_smooth: np.ndarray,
    template_path_index_direction_i: List[int],
):
    """

    Parameters
    ----------
    v1
    v2
    G
    vertices
        Vertex coordinates from cortical model.
    vertices_smooth
    template_path_index_direction_i

    Returns
    -------

    """
    v2_source_ind = template_path_index_direction_i[v2]
    coord_next = vertices[v2_source_ind]
    coord_smooth_next = vertices_smooth[v2_source_ind]
    forward_model_next = G[:, v2_source_ind]
    v1 += 1
    v2 += 1

    return v1, v2, coord_next, coord_smooth_next, forward_model_next


def create_wave_cortical_paths(
    template_path_index_array: Dict[int, List[int]],
    distance_to_next_source: Dict[int, List[int]],
    wave_generation_params: Dict[str, Any],
    start_source_index: int,
    vertices: np.ndarray,
    vertices_smooth: np.ndarray,
    G: np.ndarray,
):
    """Calculate actual wave paths considering the speed values.

    Parameters
    ----------
    template_path_index_array
    distance_to_next_source
    wave_generation_params
    start_source_index
    vertices
        Vertex coordinates from cortical model.
    vertices_smooth
    G

    Returns
    -------

    """
    speed_list = wave_generation_params["speeds"]
    wave_duration_sec = wave_generation_params["duration"]
    sampling_frequency = wave_generation_params["Fs"]

    one_step_duration_sec = 1 / sampling_frequency

    channels_num = G.shape[0]

    speed_num = len(speed_list)  # number of propagation speeds
    # number of observations
    time_points_num = int(sampling_frequency * wave_duration_sec) + 1

    # number of propagation directions
    propagation_direction_num = len(template_path_index_array.keys())

    path_coordinates_array = np.zeros([propagation_direction_num, speed_num, time_points_num, 3])
    path_coordinates_smooth_array = np.zeros([propagation_direction_num, speed_num, time_points_num, 3])
    forward_model_updated = np.zeros([propagation_direction_num, speed_num, time_points_num, channels_num])

    propagation_direction_coord = np.zeros([propagation_direction_num, speed_num, 3])
    propagation_direction_coord_smooth = np.zeros([propagation_direction_num, speed_num, 3])
    propagation_direction_coord_pca = np.zeros([propagation_direction_num, speed_num, 3])

    for direction_i in range(propagation_direction_num):
        for speed_i, speed in enumerate(speed_list):
            # distance that wave travels in one step
            one_step_distance = (speed * one_step_duration_sec)

            # initialization of coordinates with first source
            coord_previous = vertices[start_source_index]
            coord_smooth_previous = vertices_smooth[start_source_index]
            forward_model_previous = G[:, start_source_index]

            path_coordinates_array[direction_i, speed_i, 0, :] = coord_previous
            path_coordinates_smooth_array[direction_i, speed_i, 0, :] = coord_smooth_previous
            forward_model_updated[direction_i, speed_i, 0, :] = forward_model_previous

            distance_to_next_source_direction_i = distance_to_next_source[direction_i]
            template_path_index_direction_i = template_path_index_array[direction_i]
            distance_residual = 0
            v1 = 0
            v2 = 1
            for time_i in range(1, time_points_num):
                if one_step_distance < distance_residual:
                    (
                        distance_residual,
                        coord_next,
                        coord_smooth_next,
                        forward_model_next,
                    ) = one_step_lower_than_residual(
                        v2=v2,
                        G=G,
                        vertices=vertices,
                        vertices_smooth=vertices_smooth,
                        distance_residual=distance_residual,
                        one_step_distance=one_step_distance,
                        coord_previous=coord_previous,
                        coord_smooth_previous=coord_smooth_previous,
                        forward_model_previous=forward_model_previous,
                        template_path_index_direction_i=template_path_index_direction_i,
                    )
                elif one_step_distance > distance_residual:
                    (
                        v1,
                        v2,
                        distance_residual,
                        coord_next,
                        coord_smooth_next,
                        forward_model_next,
                    ) = one_step_higher_than_residual(
                        v1=v1,
                        v2=v2,
                        G=G,
                        vertices=vertices,
                        vertices_smooth=vertices_smooth,
                        distance_residual=distance_residual,
                        one_step_distance=one_step_distance,
                        template_path_index_direction_i=template_path_index_direction_i,
                        distance_to_next_source_direction_i=distance_to_next_source_direction_i,
                    )
                else:
                    (
                        v1,
                        v2,
                        coord_next,
                        coord_smooth_next,
                        forward_model_next,
                    ) = one_step_equals_to_residual(
                        v1=v1,
                        v2=v2,
                        G=G,
                        vertices=vertices,
                        vertices_smooth=vertices_smooth,
                        template_path_index_direction_i=template_path_index_direction_i,
                    )

                path_coordinates_array[direction_i, speed_i, time_i, :] = coord_next
                path_coordinates_smooth_array[direction_i, speed_i, time_i, :] = coord_smooth_next
                forward_model_updated[direction_i, speed_i, time_i, :] = forward_model_next

                coord_previous = coord_next
                coord_smooth_previous = coord_smooth_next
                forward_model_previous = forward_model_next

            propagation_direction_coord[direction_i, speed_i, :] = (
                path_coordinates_array[direction_i, speed_i, -1, :]
                - path_coordinates_array[direction_i, speed_i, 0, :]
            )

            propagation_direction_coord_smooth[direction_i, speed_i, :] = (
                path_coordinates_smooth_array[direction_i, speed_i, -1, :]
                - path_coordinates_smooth_array[direction_i, speed_i, 0, :]
            )

            [u, _, _] = np.linalg.svd(path_coordinates_array[direction_i, speed_i, :, :])
            propagation_direction_coord_pca[direction_i, speed_i, :] = (
                u[:, 1].T @ path_coordinates_array[direction_i, speed_i, :, :]
            )

    return (
        path_coordinates_array,
        path_coordinates_smooth_array,
        forward_model_updated,
        propagation_direction_coord,
        propagation_direction_coord_smooth,
        propagation_direction_coord_pca,
    )


def create_time_series_for_waves(
    wave_generation_params: Dict[str, Any], plot_time_series: bool = False
):
    """Calculate time series vectors for waves.

    Parameters
    ----------
    wave_generation_params : Dict[str, Any]
        Wave parameters (duration, sampling_frequency).
    plot_time_series : bool = False
        If True, plot time series on graph.

    Returns
    -------
    wave_time_series : np.ndarray
        Time series for waves [observed_source_num x time_points_num * 2].
        Assume that observed_source_num = time_points_num, as
        each observation corresponds to one source on the wave path.
        Each row corresponds to one source on the wave path.
        Time series length is twice as long as the number of observations.
    """
    wave_duration_sec = wave_generation_params["duration"]
    sampling_frequency = wave_generation_params["Fs"]
    # observations number
    time_points_num = int(sampling_frequency * wave_duration_sec) + 1

    step_value = 1 / 100
    time_points_list = np.arange(0, time_points_num * 1 * step_value, step=step_value)
    time_points_list_double = np.arange(0, time_points_num * 2 * step_value, step=step_value)

    time_points_grid = (
        np.tile(time_points_list_double, (time_points_num, 1))
        - np.tile(time_points_list.T, (time_points_num * 2, 1)).T
    )
    wave_time_series = (
        np.sin(10 * np.pi * time_points_grid)
        * np.exp(-10 * (2 * time_points_grid + 0.2) ** 2)
    )

    for time_i in range(time_points_num):
        wave_time_series[time_i, :time_i] = np.zeros(time_i)

    if plot_time_series:
        plt.figure()
        plt.plot(time_points_list_double, wave_time_series[0], "r", lw=3)
        plt.plot(time_points_list_double, wave_time_series.T[:, 1:], "k")
        plt.xlabel("Time, ms")
        plt.ylabel("Amplitude, a.u.")
        plt.title("Wave time series")
        plt.legend(["First source"])
        plt.grid()
    return wave_time_series


def project_wave_on_sensors(
    wave_time_series,
    wave_generation_params,
    add_spherical_wave,
    propagation_direction_num,
    channels_num,
    forward_model_updated,
):
    """

    Parameters
    ----------
    wave_time_series
    wave_generation_params
    add_spherical_wave
    propagation_direction_num
    channels_num
    forward_model_updated

    Returns
    -------

    """
    speed_list = wave_generation_params["speeds"]
    speed_num = len(speed_list)

    wave_duration_sec = wave_generation_params["duration"]
    sampling_frequency = wave_generation_params["Fs"]
    # number of time points to generate
    time_points_num = int(sampling_frequency * wave_duration_sec) + 1

    total_duration = wave_time_series.shape[1]
    if add_spherical_wave:
        waves_on_sensors = np.zeros(
            [propagation_direction_num + 1, speed_num, channels_num, total_duration]
        )
    else:
        waves_on_sensors = np.zeros(
            [propagation_direction_num, speed_num, channels_num, total_duration]
        )
    for speed_i in range(speed_num):
        for direction_i in range(propagation_direction_num):
            forward_model_submatrix = np.zeros([channels_num, time_points_num])
            for time_i in range(time_points_num):
                forward_model_submatrix[:, time_i] = forward_model_updated[direction_i, speed_i, time_i, :]
            waves_on_sensors[direction_i, speed_i, :, :] = forward_model_submatrix @ wave_time_series
    if add_spherical_wave:
        for speed_i in range(speed_num):
            for direction_i in range(propagation_direction_num):
                waves_on_sensors[propagation_direction_num, speed_i, :, :] = (
                    waves_on_sensors[propagation_direction_num, speed_i, :, :]
                    + waves_on_sensors[direction_i, speed_i, :, :]
                )

            waves_on_sensors[propagation_direction_num, speed_i, :, :] = (
                waves_on_sensors[propagation_direction_num, speed_i, :, :]
                / propagation_direction_num
            )
    return waves_on_sensors


def create_waves_on_sensors(
    vertices: np.ndarray,
    vertices_smooth: np.ndarray,
    vert_conn: csc.csc_matrix,
    vert_normals: np.ndarray,
    G: np.ndarray,
    wave_generation_params: Dict[str, Any],
    start_source_index: int,
    add_spherical_wave: bool = False,
    plot_time_series: bool = False,
):
    """Create basis wave set.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates from cortical model [sources_num x 3].
    vertices_smooth
    vert_conn : csc.csc_matrix
        Sparse matrix with vertex connections [sources_num x sources_num].
    vert_normals : np.ndarray
        Normal vector coordinates for each source [sources_num x 3].
    G : np.ndarray
        Forward model matrix [channel_num x source_num].
    wave_generation_params : Dict[str, Any]
        Wave parameters.
    start_source_index : int
        Index of the wave starting source.
    add_spherical_wave : bool = False
        If True, spherical wave is added to the set.
    plot_time_series : bool = False
        If True, plot the wave time series.

    Returns
    -------
    waves_on_sensors : np.ndarray
        Matrix with waves projected onto sensors:
        [direction_num x speed_num x channel_num x duration * 2].
    direction_final : np.ndarray
        Propagation direction coordinates [direction_num x speed_num x 3].
    direction_final_smooth : np.ndarray
        Propagation direction coordinates in smooth model [direction_num x speed_num x 3].
    direction_pca : np.ndarray
        Propagation direction coordinates if apply pca to all directions [direction_num x speed_num x 3].
    """
    channels_num = G.shape[0]
    nearest_neighbour_list = vert_conn[start_source_index, :].nonzero()[1]
    propagation_direction_num = len(nearest_neighbour_list)

    template_path_index_array, distance_to_next_source = create_template_paths(
        wave_generation_params=wave_generation_params,
        start_source_index=start_source_index,
        vertices=vertices,
        vert_conn=vert_conn,
        vert_normals=vert_normals,
    )

    (
        path_coordinates_array,
        path_coordinates_smooth_array,
        forward_model_updated,
        direction_final,
        direction_final_smooth,
        direction_pca,
    ) = create_wave_cortical_paths(
        template_path_index_array=template_path_index_array,
        distance_to_next_source=distance_to_next_source,
        wave_generation_params=wave_generation_params,
        start_source_index=start_source_index,
        vertices=vertices,
        vertices_smooth=vertices_smooth,
        G=G,
    )

    wave_time_series = create_time_series_for_waves(
        wave_generation_params=wave_generation_params, plot_time_series=plot_time_series
    )

    waves_on_sensors = project_wave_on_sensors(
        wave_time_series=wave_time_series,
        wave_generation_params=wave_generation_params,
        add_spherical_wave=add_spherical_wave,
        propagation_direction_num=propagation_direction_num,
        channels_num=channels_num,
        forward_model_updated=forward_model_updated,
    )

    return waves_on_sensors, direction_final, direction_final_smooth, direction_pca
