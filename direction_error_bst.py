import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn import metrics
from typing import List, Dict, Any
from collections import defaultdict

from create_waves_on_sensors import create_waves_on_sensors
from create_blob_on_sensors import create_blob_on_sensors
from generate_brain_noise import generate_brain_noise
from LASSO_inverse_solve import LASSO_inverse_solve


def find_close_sources(
        source_index: int,
        vertices: np.ndarray,
        vertices_dense: np.ndarray,
        spatial_jitter: float,
        area_radius: float = 0.003
):
    """Compute the distance from one source from sparse cortical model to all sources from the dense cortical model.
    Select close sources disturbed with spatial jitter: distance must be higher than spatial_jitter and lower than
    spatial jitter + area_radius (in meters).

    Parameters
    ----------
    source_index : int
        Index of fixed source from sparse cortical model.
    vertices : np.ndarray
        Vertex coordinates for sparse model [n_sources_sparce x 3].
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
    starting_source_coordinates_repeated = (
        np.repeat(
            vertices[source_index, np.newaxis],
            vertices_dense.shape[0],
            axis=0,
        )
    )
    distance_vector = np.linalg.norm(
        starting_source_coordinates_repeated - vertices_dense,
        axis=1
    )

    close_sources_index_list = (
        np.where((distance_vector > spatial_jitter) & (distance_vector <= (spatial_jitter + area_radius)))[0]
    )

    return close_sources_index_list


def calculate_direction_error(
        direction_estimated: np.ndarray,
        direction_generated: np.ndarray
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
    error = (
            1
            - abs(
                direction_estimated @ direction_generated
                / np.linalg.norm(direction_estimated)
                / np.linalg.norm(direction_generated)
            )
    )
    return error


def add_brain_noise_to_signal(
        signal_on_sensors: np.ndarray,
        brain_noise: np.ndarray,
        snr: float
):
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
        Signal of interest mixed with brain noise for fixed SNR level [n_channles x total_duration].
    """
    signal_on_sensors_normalized = signal_on_sensors / np.linalg.norm(signal_on_sensors)
    brain_noise_normalized = brain_noise / np.linalg.norm(brain_noise)
    data = snr * signal_on_sensors_normalized + brain_noise_normalized
    return data


def direction_error_bst(
        data_dir: str,
        channel_type: str,
        wave_generation_params: Dict[str, Any],
        snr_list: List[float],
        spatial_jitter_list: List[float],
        simulation_number: int = 100,
):
    """Monte Carlo simulations.
    Function calculates speed and direction error as a function of spatial jitter and SNR.
    All cortical models and forward operators are calculated in Brainstorm.

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
    wave_generation_params : Dict[str, Any]
        Wave modeling parameters.
    snr_list : List[float]
        List of SNR values to consider.
    spatial_jitter_list : List[float]
        List of lower bound values (in m) for spatial jitter. Spatial jitter is distributed uniformly
        between [jitter[i], jitter[i] + 0.003].
    simulation_number : int
        Simulation number for one class.

    Returns
    -------
    auc :
        AUC values for all SNR levels.
    speed_error :
        Difference between simulated and detected speeds in m/s.
    direction_error :
        1 - correlation between generated and detected directions.
    """
    (
        G_dense,
        G,
        cortex_dense,
        cortex,
        cortex_smooth_dense,
        cortex_smooth,
        vertices,
        vertices_dense
    ) = load_input_data(data_dir=data_dir, channel_type=channel_type)

    sparse_source_number = G.shape[1]
    speed_number = len(wave_generation_params["speeds"])

    # wave duration in ms
    total_duration = int(wave_generation_params["duration"] * wave_generation_params["Fs"] + 1) * 2

    spatial_jitter_num = len(spatial_jitter_list)
    snr_num = len(snr_list)

    spatial_error = np.zeros([spatial_jitter_num, snr_num, simulation_number])
    speed_simulated_array = np.zeros([spatial_jitter_num, snr_num, simulation_number], dtype=int)
    speed_estimated_array = np.zeros([spatial_jitter_num, snr_num, simulation_number], dtype=int)

    direction_error = np.zeros([spatial_jitter_num, snr_num, simulation_number])
    direction_error_smooth = np.zeros([spatial_jitter_num, snr_num, simulation_number])
    direction_error_pca = np.zeros([spatial_jitter_num, snr_num, simulation_number])

    y_true = [1] * simulation_number + [0] * simulation_number  # true labels
    roc_parameters = defaultdict(dict)

    for spatial_jitter_i, spatial_jitter in enumerate(spatial_jitter_list):
        for snr_i, snr in enumerate(snr_list):

            starting_source_sparse_list = []  # assumed starting source from the sparse model
            starting_source_dense_list = []  # true starting source from the dense model
            r_squared_per_simulation = []  # R-squared values for all simulations

            # true directions for each simulation
            direction_simulated = np.zeros([simulation_number, 3])
            direction_simulated_smooth = np.zeros([simulation_number, 3])
            direction_simulated_pca = np.zeros([simulation_number, 3])

            # brain noise array
            brain_noise_array = np.zeros([G_dense.shape[0], total_duration, simulation_number])

            # first `simulation_number` trials are traveling waves
            for simulation_i in range(simulation_number):

                # random starting source from sparse cortical model
                starting_source_sparse_list.append(np.random.randint(sparse_source_number))

                # find close sources from dense_model
                close_source_index_list = find_close_sources(
                    source_index=starting_source_sparse_list[simulation_i],
                    vertices=vertices,
                    vertices_dense=vertices_dense,
                    spatial_jitter=spatial_jitter,
                    area_radius=0.003
                )

                if close_source_index_list.size == 0:
                    continue

                # pick randomly new starting source from close sources of dense model
                starting_source_dense_list.append(np.random.choice(close_source_index_list))
                spatial_error[spatial_jitter_i, snr_i, simulation_i] = np.linalg.norm(
                    vertices_dense[starting_source_dense_list[simulation_i]]
                    - vertices[starting_source_sparse_list[simulation_i]]
                )

                # calculate true traveling wave set on dense cortical grid
                [
                    waves_on_sensors,
                    _,
                    _,
                    propagation_direction_list,
                    propagation_direction_smooth_list,
                    propagation_direction_pca_list,
                ] = create_waves_on_sensors(
                    cortex=cortex_dense,
                    cortex_smooth=cortex_smooth_dense,
                    params=wave_generation_params,
                    G=G_dense,
                    start_point=starting_source_dense_list[simulation_i],
                    spherical=False
                )

                # select randomly one wave from the set (speed and direction)
                speed_simulated_index = np.random.randint(speed_number)
                speed_simulated_array[spatial_jitter_i, snr_i, simulation_i] = speed_simulated_index

                direction_number = waves_on_sensors.shape[0]
                direction_simulated_index = np.random.randint(direction_number)
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
                brain_noise_array[:, :, simulation_i] = generate_brain_noise(
                    G=G_dense,
                    time_point_number=total_duration
                )

                data = add_brain_noise_to_signal(
                    signal_on_sensors=wave_selected,
                    brain_noise=brain_noise_array[:, :, simulation_i],
                    snr=snr
                )

                # Generate basis waves using sparse cortical model
                # starting source is the initially picked point (with spatial error)
                [
                    waves_on_sensors,
                    _,
                    _,
                    propagation_direction_list,
                    propagation_direction_smooth_list,
                    propagation_direction_pca_list,
                ] = create_waves_on_sensors(
                    cortex=cortex,
                    cortex_smooth=cortex_smooth,
                    params=wave_generation_params,
                    G=G,
                    start_point=starting_source_sparse_list[simulation_i],
                    spherical=False
                )

                # Solve the LASSO problem without intercept
                [score, _, best_coefficients, _, speed_estimated_index] = (
                    LASSO_inverse_solve(data=data, waves=waves_on_sensors, fit_intercept=False)
                )
                r_squared_per_simulation.append(score)
                speed_estimated_array[spatial_jitter_i, snr_i, simulation_i] = speed_estimated_index

                # calculate direction error
                direction_estimated_index = np.argmax(best_coefficients)
                direction_estimated = propagation_direction_list[direction_estimated_index, speed_estimated_index, :]
                direction_generated = direction_simulated[simulation_i, :]

                direction_error[spatial_jitter_i, snr_i, simulation_i] = calculate_direction_error(
                    direction_estimated=direction_estimated,
                    direction_generated=direction_generated
                )

                direction_smooth_estimated = (
                    propagation_direction_smooth_list[direction_estimated_index, speed_estimated_index, :]
                )
                direction_smooth_generated = direction_simulated_smooth[simulation_i, :]
                direction_error_smooth[spatial_jitter_i, snr_i, simulation_i] = calculate_direction_error(
                    direction_estimated=direction_smooth_estimated,
                    direction_generated=direction_smooth_generated
                )

                direction_pca_estimated = (
                    propagation_direction_pca_list[direction_estimated_index, speed_estimated_index, :]
                )
                direction_pca_generated = direction_simulated_pca[simulation_i, :]
                direction_error_pca[spatial_jitter_i, snr_i, simulation_i] = calculate_direction_error(
                    direction_estimated=direction_pca_estimated,
                    direction_generated=direction_pca_generated
                )

                print(f"{simulation_i + 1} out of {simulation_number} completed (trials with waves)")

            # next `simulation_number` trials are static oscillating blobs
            for simulation_i in range(simulation_number, 2 * simulation_number):
                starting_source_dense_index = starting_source_dense_list[simulation_i - simulation_number]
                starting_source_sparse_index = starting_source_sparse_list[simulation_i - simulation_number]

                [oscillating_blob_on_sensors, _] = create_blob_on_sensors(
                    cortex=cortex_dense,
                    G=G_dense,
                    start_point=starting_source_dense_index,
                    T=total_duration
                )

                data = add_brain_noise_to_signal(
                    signal_on_sensors=oscillating_blob_on_sensors,
                    brain_noise=brain_noise_array[:, :, simulation_i - simulation_number],
                    snr=snr
                )

                [waves_on_sensors, *_] = create_waves_on_sensors(
                    cortex=cortex,
                    cortex_smooth=cortex_smooth,
                    params=wave_generation_params,
                    G=G,
                    start_point=starting_source_sparse_index,
                    spherical=False
                )

                [score, *_] = LASSO_inverse_solve(
                    data=data,
                    waves=waves_on_sensors,
                    fit_intercept=False
                )
                r_squared_per_simulation.append(score)

                print(f"{simulation_i - simulation_number + 1} out of {simulation_number} completed (trials w/o waves)")

            # calculate ROC AUC, FPR, TPR
            y_score = r_squared_per_simulation
            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
            roc_parameters[(spatial_jitter, snr)]['fpr'] = fpr
            roc_parameters[(spatial_jitter, snr)]['tpr'] = tpr
            roc_parameters[(spatial_jitter, snr)]['auc'] = metrics.roc_auc_score(y_true, y_score)

    return (
        roc_parameters,
        direction_error,
        direction_error_smooth,
        direction_error_pca,
        speed_simulated_array,
        speed_estimated_array,
    )


def plot_roc(
    spatial_jitter_list: List[float],
    snr_list: List[float],
    roc_parameters: defaultdict
):

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
                label=f"ROC curve for SNR = {snr}, (AUC = {auc})"
            )

        plt.title(f"Mean spatial jitter = {(spatial_jitter + 0.003 / 2) * 1000} mm")
        plt.legend(loc="lower right")
        plt.show()


#     plt.figure()
#     plt.subplot(1, 3, 1)
#     for j in range(spatial_jitter_num):
#         plt.plot(snr_list, np.mean(direction_error[j, :, :], axis=1), "o-")
#         plt.title("Direction detection error (with curvature)")
#         plt.ylim([0, 1])
#         plt.xlabel("SNR")
#         plt.ylabel("1 - correlation between generated and detected")
#     plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)
#     plt.subplot(1, 3, 2)
#     for j in range(spatial_jitter_num):
#         plt.plot(snr_list, np.mean(direction_error_smooth[j, :, :], axis=1), "o-")
#         plt.title("Direction detection error (on smooth cortex)")
#         plt.ylim([0, 1])
#         plt.xlabel("SNR")
#         plt.ylabel("1 - correlation between generated and detected")
#     plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)
#     plt.subplot(1, 3, 3)
#     for j in range(spatial_jitter_num):
#         plt.plot(snr_list, np.mean(direction_error_pca[j, :, :], axis=1), "o-")
#         plt.title("Direction detection error (main direction)")
#         plt.xlabel("SNR")
#         plt.ylim([0, 1])
#         plt.ylabel("Error between detected and generated speeds in m/s")
#     plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)
#
#     plt.figure()
#     k = 1
#     for j in range(spatial_jitter_num):
#         for s in range(speed_generated.shape[0]):
#             jitter_1 = 0.2 * np.random.rand(speed_estimated.shape[2])
#             jitter_2 = 0.2 * np.random.rand(speed_estimated.shape[2])
#             plt.subplot(spatial_jitter_num, snr_num, k)
#             plt.scatter(
#                 speed_generated[j, s, :] + jitter_1, speed_estimated[j, s, :] + jitter_2
#             )
#             plt.xlabel("True speed, m/s")
#             plt.ylabel("Estimated speed, m/s")
#             plt.plot(np.arange(-1, 10, 0.1), np.arange(-1, 10, 0.1), "--")
#             plt.xlim([-1, 10])
#             plt.ylim([-1, 10])
#             plt.xticks(range(10), wave_generation_params["speeds"])
#             plt.yticks(range(10), wave_generation_params["speeds"])
#             plt.title(
#                 "SNR level = "
#                 + str(snr_list[s])
#                 + ", Mean spatial jitter = "
#                 + str((spatial_jitter_list[j] + 0.0015) * 1000)
#                 + " mm"
#             )
#             k += 1
