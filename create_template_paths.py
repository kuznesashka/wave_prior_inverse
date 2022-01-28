import numpy as np
from typing import List, Dict, Any
import scipy.sparse.csc as csc
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
