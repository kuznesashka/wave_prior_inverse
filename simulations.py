import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.csc as csc

from sklearn import metrics
from typing import List, Dict, Tuple, Any
from collections import defaultdict

from create_waves_on_sensors import create_waves_on_sensors
from create_blob_on_sensors import create_blob_on_sensors
from generate_brain_noise import generate_brain_noise
from lasso_inverse_solve import lasso_inverse_solve


def find_close_sources(
        source_index: int,
        vertices: np.ndarray,
        vertices_dense: np.ndarray,
        spatial_jitter: float,
        area_radius: float = 0.003,
) -> np.ndarray:
    """Compute the distance from one source from sparse cortical model to all sources from the dense cortical model.
    Select close sources disturbed with spatial jitter: distance must be higher than spatial_jitter and lower than
    spatial jitter + area_radius (in meters).

    Parameters
    ----------
    source_index : int
        Index of fixed source from sparse cortical model.
    vertices : np.ndarray
        Vertex coordinates for sparse model [n_sources_sparse x 3].
    vertices_dense : np.ndarray
        Vertex coordinates for dense model [n_sources_dense x 3].
    spatial_jitter : float
        Spatial error (in meters) introduced while selecting closest sources from dense cortical model to one
        source from sparse cortical model.
    area_radius : float
        Radius (in meters) of area considered as close.

    Returns
    -------
    close_sources_index_list : np.ndarray
        Indices of close sources.
    """
    starting_source_coordinates_repeated = np.repeat(
        vertices[source_index, np.newaxis],
        vertices_dense.shape[0],
        axis=0,
    )
    distance_vector = np.linalg.norm(starting_source_coordinates_repeated - vertices_dense, axis=1)

    close_sources_index_list = (
        np.where((distance_vector > spatial_jitter) & (distance_vector <= (spatial_jitter + area_radius)))[0]
    )
    return close_sources_index_list


def calculate_direction_error(
        direction_estimated: np.ndarray, direction_generated: np.ndarray
) -> float:
    """Calculates error between estimated (a) and true (b) propagation directions as 1 - cos(a, b).

    Parameters
    ----------
    direction_estimated : np.ndarray
        Estimated propagation direction coordinates [1 x 3].
    direction_generated : np.ndarray
        True propagation direction coordinates [1 x 3].

    Returns
    -------
    error : float
        Error as 1 - cos(a, b).
    """
    error = 1 - abs(
        direction_estimated
        @ direction_generated
        / np.linalg.norm(direction_estimated)
        / np.linalg.norm(direction_generated)
    )
    return error


def add_brain_noise_to_signal(
        signal_on_sensors: np.ndarray, brain_noise: np.ndarray, snr: float
) -> np.ndarray:
    """Add brain noise to generated signal.

    Parameters
    ----------
    signal_on_sensors : np.ndarray
        Signal of interest [n_channels x total_duration].
    brain_noise : np.ndarray
        Brain noise [n_channels x total_duration].
    snr : float
        SNR level.

    Returns
    -------
    data : np.ndarray
        Signal of interest mixed with brain noise for fixed SNR level [n_channels x total_duration].
    """
    signal_on_sensors_normalized = signal_on_sensors / np.linalg.norm(signal_on_sensors)
    brain_noise_normalized = brain_noise / np.linalg.norm(brain_noise)
    data = snr * signal_on_sensors_normalized + brain_noise_normalized
    return data


def pick_random_source(vertices: np.ndarray, distance_to_midline: float = 0.02) -> int:
    """Select one random source not closer than `distance_to_midline` (in m) to midline.

    Parameters
    ----------
    vertices : np.ndarray
    distance_to_midline : float

    Returns
    -------
    picked_source_ind : int
    """
    x_axis = 1
    far_from_midline_mask = np.abs(vertices[:, x_axis]) >= distance_to_midline
    sources_far_from_midline = np.where(far_from_midline_mask)[0]
    picked_source_ind = np.random.choice(sources_far_from_midline)
    return picked_source_ind


def pick_source_counterpart(
        vertices,
        vertices_dense,
        source_index,
        spatial_jitter
):
    # find close sources from dense_model
    close_source_index_list = find_close_sources(
        source_index=source_index,
        vertices=vertices,
        vertices_dense=vertices_dense,
        spatial_jitter=spatial_jitter,
        area_radius=0.003,
    )

    if close_source_index_list.size == 0:
        return -1, -1

    # pick randomly new starting source from close sources of dense model
    starting_source_dense = np.random.choice(close_source_index_list)
    spatial_error = np.linalg.norm(
        vertices_dense[starting_source_dense]
        - vertices[source_index]
    )
    return starting_source_dense, spatial_error


def calculate_direction_speed_error(
        G: np.ndarray,
        G_dense: np.ndarray,
        vertices: np.ndarray,
        vertices_dense: np.ndarray,
        vertices_smooth: np.ndarray,
        vertices_smooth_dense: np.ndarray,
        vert_conn: csc.csc_matrix,
        vert_conn_dense: csc.csc_matrix,
        vert_normals: np.ndarray,
        vert_normals_dense: np.ndarray,
        wave_generation_params: Dict[str, Any],
        snr_list: List[float],
        spatial_jitter_list: List[float],
        simulation_num: int = 100,
        add_spherical_wave: bool = False,
        plot_wave_time_series: bool = False,
        path_length_for_blob: int = 20,
        plot_blob_time_series: bool = False,
        distance_to_midline: float = 0.02,
        lasso_fit_intercept: bool = False
) -> Tuple:
    """Monte Carlo simulations.
    Function calculates speed and direction error as a function of spatial jitter and SNR.
    All cortical models and forward operators are calculated in Brainstorm.

    Parameters
    ----------
    G : np.ndarray
    G_dense : np.ndarray
    vertices : np.ndarray
    vertices_dense : np.ndarray
    vertices_smooth : np.ndarray
    vertices_smooth_dense : np.ndarray
    vert_conn : csc.csc_matrix
    vert_conn_dense : csc.csc_matrix
    vert_normals : np.ndarray
    vert_normals_dense : np.ndarray
    wave_generation_params : Dict[str, Any]
    snr_list : List[float]
    spatial_jitter_list : List[float]
    simulation_num : int = 100
    add_spherical_wave : bool = False
    plot_wave_time_series : bool = False
    path_length_for_blob : int = 20
    plot_blob_time_series : bool = False
    distance_to_midline : float = 0.02
    lasso_fit_intercept : bool = False

    Returns
    -------
    roc_parameters : Dict[int, Any]
    direction_simulated_array : np.ndarray
    direction_error : np.ndarray
    direction_error_smooth : np.ndarray
    direction_error_pca : np.ndarray
    speed_simulated_array : np.ndarray
    speed_estimated_array : np.ndarray
    spatial_error : np.ndarray
    """
    channel_num = G.shape[0]

    speed_number = len(wave_generation_params["speeds"])
    duration_sec = wave_generation_params["duration"]
    sampling_frequency = wave_generation_params["Fs"]
    total_duration = int(duration_sec * sampling_frequency + 1) * 2  # wave duration (time samples)

    spatial_jitter_num = len(spatial_jitter_list)
    snr_num = len(snr_list)

    spatial_error = np.zeros([spatial_jitter_num, snr_num, simulation_num])
    speed_simulated_array = np.zeros([spatial_jitter_num, snr_num, simulation_num], dtype=int)
    speed_estimated_array = np.zeros([spatial_jitter_num, snr_num, simulation_num], dtype=int)

    direction_simulated_array = np.zeros([spatial_jitter_num, snr_num, simulation_num], dtype=int)
    direction_error = np.zeros([spatial_jitter_num, snr_num, simulation_num])
    direction_error_smooth = np.zeros([spatial_jitter_num, snr_num, simulation_num])
    direction_error_pca = np.zeros([spatial_jitter_num, snr_num, simulation_num])

    y_true = [1] * simulation_num + [0] * simulation_num  # true labels
    roc_parameters = defaultdict(dict)

    for spatial_jitter_i, spatial_jitter in enumerate(spatial_jitter_list):
        for snr_i, snr in enumerate(snr_list):
            print(f"calculations for SNR = {snr} out of {snr_list}")
            # assumed starting source from the sparse model
            starting_source_sparse_list = []
            # true starting source from the dense model
            starting_source_dense_list = []
            # R-squared values for all simulations
            r_squared_per_wave_simulation = []
            r_squared_per_blob_simulation = []

            direction_simulated = np.zeros([simulation_num, 3])
            direction_simulated_smooth = np.zeros([simulation_num, 3])
            direction_simulated_pca = np.zeros([simulation_num, 3])

            data_wave = np.zeros([simulation_num, channel_num, total_duration])
            data_blob = np.zeros([simulation_num, channel_num, total_duration])

            for simulation_i in range(simulation_num):
                # random starting source from sparse cortical model
                source = pick_random_source(vertices=vertices, distance_to_midline=distance_to_midline)
                starting_source_sparse_list.append(source)

                # pick randomly new starting source from close sources of dense model
                source_dense, error = pick_source_counterpart(
                    vertices=vertices,
                    vertices_dense=vertices_dense,
                    source_index=starting_source_sparse_list[simulation_i],
                    spatial_jitter=spatial_jitter
                )

                if source_dense == -1:
                    continue

                starting_source_dense_list.append(source_dense)
                spatial_error[spatial_jitter_i, snr_i, simulation_i] = error

                # calculate true traveling wave set on dense cortical grid
                (
                    waves_on_sensors,
                    propagation_direction_list,
                    propagation_direction_smooth_list,
                    propagation_direction_pca_list,
                ) = create_waves_on_sensors(
                    vertices=vertices_dense,
                    vertices_smooth=vertices_smooth_dense,
                    vert_conn=vert_conn_dense,
                    vert_normals=vert_normals_dense,
                    G=G_dense,
                    wave_generation_params=wave_generation_params,
                    start_source_index=starting_source_dense_list[simulation_i],
                    add_spherical_wave=add_spherical_wave,
                    plot_time_series=plot_wave_time_series,
                )

                # select randomly one wave from the set (speed and direction)
                speed_simulated_index = np.random.randint(speed_number)
                speed_simulated_array[spatial_jitter_i, snr_i, simulation_i] = speed_simulated_index

                direction_number = waves_on_sensors.shape[0]
                direction_simulated_index = np.random.randint(direction_number)
                direction_simulated_array[spatial_jitter_i, snr_i, simulation_i] = direction_simulated_index

                direction_simulated[simulation_i, :] = (
                    propagation_direction_list[direction_simulated_index, speed_simulated_index, :]
                )

                direction_simulated_smooth[simulation_i, :] = (
                    propagation_direction_smooth_list[direction_simulated_index, speed_simulated_index, :]
                )

                direction_simulated_pca[simulation_i, :] = (
                    propagation_direction_pca_list[direction_simulated_index, speed_simulated_index, :]
                )

                wave_selected = waves_on_sensors[direction_simulated_index, speed_simulated_index, :, :]
                brain_noise = generate_brain_noise(G=G_dense, time_point_number=total_duration)

                data_wave[simulation_i, :, :] = add_brain_noise_to_signal(
                    signal_on_sensors=wave_selected,
                    brain_noise=brain_noise,
                    snr=snr,
                )

                # Generate basis waves using sparse cortical model
                # starting source is the initially picked point (with spatial error)
                (
                    waves_on_sensors,
                    propagation_direction_list,
                    propagation_direction_smooth_list,
                    propagation_direction_pca_list,
                ) = create_waves_on_sensors(
                    vertices=vertices,
                    vertices_smooth=vertices_smooth,
                    vert_conn=vert_conn,
                    vert_normals=vert_normals,
                    G=G,
                    wave_generation_params=wave_generation_params,
                    start_source_index=starting_source_sparse_list[simulation_i],
                    add_spherical_wave=add_spherical_wave,
                    plot_time_series=plot_wave_time_series,
                )

                score, speed_estimated_index, _, coefficients, _ = lasso_inverse_solve(
                    signal_data=data_wave[simulation_i, :, :],
                    wave_data=waves_on_sensors,
                    fit_intercept=lasso_fit_intercept
                )

                r_squared_per_wave_simulation.append(score)
                speed_estimated_array[spatial_jitter_i, snr_i, simulation_i] = speed_estimated_index

                # calculate direction error
                direction_estimated_index = np.argmax(coefficients)
                direction_estimated = (
                    propagation_direction_list[direction_estimated_index, speed_estimated_index, :]
                )
                direction_generated = direction_simulated[simulation_i, :]

                direction_error[spatial_jitter_i, snr_i, simulation_i] = calculate_direction_error(
                    direction_estimated=direction_estimated,
                    direction_generated=direction_generated,
                )

                direction_smooth_estimated = (
                    propagation_direction_smooth_list[direction_estimated_index, speed_estimated_index, :]
                )
                direction_smooth_generated = direction_simulated_smooth[simulation_i, :]
                direction_error_smooth[spatial_jitter_i, snr_i, simulation_i] = calculate_direction_error(
                    direction_estimated=direction_smooth_estimated,
                    direction_generated=direction_smooth_generated,
                )

                direction_pca_estimated = (
                    propagation_direction_pca_list[direction_estimated_index, speed_estimated_index, :]
                )
                direction_pca_generated = direction_simulated_pca[simulation_i, :]
                direction_error_pca[spatial_jitter_i, snr_i, simulation_i] = calculate_direction_error(
                    direction_estimated=direction_pca_estimated,
                    direction_generated=direction_pca_generated,
                )

                oscillating_blob_on_sensors, _ = create_blob_on_sensors(
                    vertices=vertices_dense,
                    vert_conn=vert_conn_dense,
                    vert_normals=vert_normals_dense,
                    G=G_dense,
                    start_source_index=source_dense,
                    total_duration=total_duration,
                    path_length=path_length_for_blob,
                    plot_time_series=plot_blob_time_series
                )

                data_blob[simulation_i, :, :] = add_brain_noise_to_signal(
                    signal_on_sensors=oscillating_blob_on_sensors,
                    brain_noise=brain_noise,
                    snr=snr,
                )

                score, *_ = lasso_inverse_solve(
                    signal_data=data_blob[simulation_i, :, :],
                    wave_data=waves_on_sensors,
                    fit_intercept=lasso_fit_intercept
                )
                r_squared_per_blob_simulation.append(score)

                print(
                    f"{simulation_i + 1} out of {simulation_num} completed"
                )

            y_score = r_squared_per_wave_simulation + r_squared_per_blob_simulation
            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
            roc_parameters[(spatial_jitter, snr)]["data_blob"] = data_blob
            roc_parameters[(spatial_jitter, snr)]["data_wave"] = data_wave
            roc_parameters[(spatial_jitter, snr)]["start_sparse"] = starting_source_sparse_list
            roc_parameters[(spatial_jitter, snr)]["start_dense"] = starting_source_dense_list
            roc_parameters[(spatial_jitter, snr)]["y_score"] = y_score
            roc_parameters[(spatial_jitter, snr)]["fpr"] = fpr
            roc_parameters[(spatial_jitter, snr)]["tpr"] = tpr
            roc_parameters[(spatial_jitter, snr)]["auc"] = metrics.roc_auc_score(y_true, y_score)

    return (
        roc_parameters,
        direction_simulated_array,
        direction_error,
        direction_error_smooth,
        direction_error_pca,
        speed_simulated_array,
        speed_estimated_array,
        spatial_error
    )


def plot_roc(
        spatial_jitter_list: List[float], snr_list: List[float], roc_parameters: defaultdict
):
    """Plot ROC curves for all spatial_jitters and SNRs.

    Parameters
    ----------
    spatial_jitter_list : List[float]
    snr_list : List[float]
    roc_parameters : defaultdict
    """
    spatial_jitter_num = len(spatial_jitter_list)

    plt.figure()
    for spatial_jitter_i, spatial_jitter in enumerate(spatial_jitter_list):
        plt.subplot(1, spatial_jitter_num, (spatial_jitter_i + 1))
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        for snr_i, snr in enumerate(snr_list):
            auc = round(roc_parameters[(spatial_jitter, snr)]["auc"], 4)
            plt.plot(
                roc_parameters[(spatial_jitter, snr)]["fpr"],
                roc_parameters[(spatial_jitter, snr)]["tpr"],
                lw=2,
                label=f"ROC curve for SNR = {snr}, (AUC = {auc})",
            )

        plt.title(f"Mean spatial jitter = {(spatial_jitter + 0.003 / 2) * 1000} mm")
        plt.legend(loc="lower right")
        plt.show()


def plot_direction_estimation_error(
        snr_list: List[float],
        spatial_jitter_list: List[float],
        direction_error: np.ndarray,
        direction_error_smooth: np.ndarray,
        direction_error_pca: np.ndarray,
):
    """Visualize error in direction estimation for different spatial jitters and SNRs.

    Parameters
    ----------
    snr_list : List[float]
    spatial_jitter_list : List[float]
    direction_error : np.ndarray
    direction_error_smooth : np.ndarray
    direction_error_pca : np.ndarray
    """
    plt.figure()
    plt.subplot(1, 3, 1)
    for spatial_jitter_i, _ in enumerate(spatial_jitter_list):
        plt.plot(snr_list, np.mean(direction_error[spatial_jitter_i, :, :], axis=1), "o-")
        plt.title("Direction detection error (with curvature)")
        plt.ylim([0, 1])
        plt.xlabel("SNR")
        plt.ylabel("1 - correlation between generated and detected")
    plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)
    plt.subplot(1, 3, 2)
    for spatial_jitter_i, _ in enumerate(spatial_jitter_list):
        plt.plot(snr_list, np.mean(direction_error_smooth[spatial_jitter_i, :, :], axis=1), "o-")
        plt.title("Direction detection error (on smooth cortex)")
        plt.ylim([0, 1])
        plt.xlabel("SNR")
        plt.ylabel("1 - correlation between generated and detected")
    plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)
    plt.subplot(1, 3, 3)
    for spatial_jitter_i, _ in enumerate(spatial_jitter_list):
        plt.plot(snr_list, np.mean(direction_error_pca[spatial_jitter_i, :, :], axis=1), "o-")
        plt.title("Direction detection error (main direction)")
        plt.xlabel("SNR")
        plt.ylim([0, 1])
        plt.ylabel("Error between detected and generated speeds in m/s")
    plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)


def plot_speed_estimation_error(
        snr_list: List[float],
        spatial_jitter_list: List[float],
        speed_simulated_array: np.ndarray,
        speed_estimated_array: np.ndarray,
        wave_generation_params: Dict[str, Any],
):
    """Visualize error in speed estimation for different spatial jitters and SNRs.

    Parameters
    ----------
    snr_list : List[float]
    spatial_jitter_list : List[float]
    speed_simulated_array : np.ndarray
    speed_estimated_array : np.ndarray
    wave_generation_params : Dict[str, Any]
    """
    speed_list = wave_generation_params["speeds"]
    spatial_jitter_num = len(spatial_jitter_list)
    snr_num = len(snr_list)
    simulation_num = speed_simulated_array.shape[2]

    plt.figure()
    k = 1
    for spatial_jitter_i, spatial_jitter in enumerate(spatial_jitter_list):
        for snr_i, snr in enumerate(snr_list):
            jitter_1 = 0.2 * np.random.rand(simulation_num)
            jitter_2 = 0.2 * np.random.rand(simulation_num)
            plt.subplot(spatial_jitter_num, snr_num, k)
            plt.scatter(
                speed_simulated_array[spatial_jitter_i, snr_i, :] + jitter_1,
                speed_estimated_array[spatial_jitter_i, snr_i, :] + jitter_2
            )
            plt.xlabel("True speed, m/s")
            plt.ylabel("Estimated speed, m/s")
            plt.plot(np.arange(-1, 10, 0.1), np.arange(-1, 10, 0.1), "--")
            plt.xlim([-1, 10])
            plt.ylim([-1, 10])
            plt.xticks(range(10), speed_list)
            plt.yticks(range(10), speed_list)
            plt.title(
                "SNR level = "
                + str(snr)
                + ", Mean spatial jitter = "
                + str((spatial_jitter + 0.0015) * 1000)
                + " mm"
            )
            k += 1
