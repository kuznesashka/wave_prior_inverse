import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn import metrics
from typing import List, Dict, Any

from create_waves_on_sensors import create_waves_on_sensors
from create_blob_on_sensors import create_blob_on_sensors
from generate_brain_noise import generate_brain_noise
from LASSO_inverse_solve import LASSO_inverse_solve


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
        - G_dense.mat (forward_operator, high source number)
        - cortes_dense.mat (cortical model, high source number)
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

    ROC curve plot
    Error plots
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

    vertices = cortex[0][1]
    vertices_dense = cortex_dense[0][1]

    T = int(wave_generation_params["duration"] * wave_generation_params["Fs"] + 1) * 2  # wave duration in ms

    spatial_jitter_num = len(spatial_jitter_list)
    snr_num = len(snr_list)
    
    speed_generated = np.zeros([spatial_jitter_num, snr_num, simulation_number], dtype=int)
    speed_estimated = np.zeros([spatial_jitter_num, snr_num, simulation_number], dtype=int)
    direction_error = np.zeros([spatial_jitter_num, snr_num, simulation_number])
    direction_error_smooth = np.zeros([spatial_jitter_num, snr_num, simulation_number])
    direction_error_pca = np.zeros([spatial_jitter_num, snr_num, simulation_number])
    spatial_error = np.zeros([spatial_jitter_num, snr_num, simulation_number])

    # ROC curve
    y_true = [1] * simulation_number + [0] * simulation_number  # true labels
    auc = np.zeros([snr_num, spatial_jitter_num])

    # ROC figure
    plt.figure()

    for i, spatial_jitter in enumerate(spatial_jitter_list):
        plt.subplot(1, spatial_jitter_num, (i + 1))
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        for j, snr in enumerate(snr_list):

            score_fit = np.zeros(2 * simulation_number)  # R-squared metrics for all simulations

            # true directions modeled
            generate_direction = np.zeros([simulation_number, 3])
            generate_direction_smooth = np.zeros([simulation_number, 3])
            generate_direction_pca = np.zeros([simulation_number, 3])

            starting_source_sparse = []  # assumed starting source from the sparse model
            starting_source_dense = []  # true starting source from the dense model

            # brain noise array
            brain_noise_norm = np.zeros([G_dense.shape[0], T, simulation_number])

            # first `simulation_number` trials are traveling waves
            for simulation_i in range(simulation_number):

                # random starting source from sparse cortical model
                starting_source_sparse.append(np.random.randint(G.shape[1]))

                # compute the distance from starting source to all sources from the dense cortical model
                starting_source_coordinates_repeated = (
                    np.repeat(
                        vertices[starting_source_sparse[simulation_i], np.newaxis],
                        vertices_dense.shape[0],
                        axis=0,
                    )
                )

                distance_vector = np.linalg.norm(starting_source_coordinates_repeated - vertices_dense, axis=1)

                # find close sources
                close_sources_index_list = np.where(
                    (distance_vector > spatial_jitter) & (distance_vector <= (spatial_jitter + 0.003))
                )[0]

                if close_sources_index_list.size == 0:
                    continue

                # pick randomly new starting source
                starting_source_dense.append(np.random.choice(close_sources_index_list))
                spatial_error[i, j, simulation_i] = np.linalg.norm(
                    vertices_dense[starting_source_dense[simulation_i]]
                    - vertices[starting_source_sparse[simulation_i]]
                )

                # calculate traveling wave
                [
                    sensor_waves,
                    _,
                    _,
                    direction,
                    direction_smooth,
                    direction_pca,
                ] = create_waves_on_sensors(
                    cortex=cortex_dense,
                    cortex_smooth=cortex_smooth_dense,
                    params=wave_generation_params,
                    G=G_dense,
                    start_point=starting_source_dense[simulation_i],
                    spherical=False
                )

                # speed for wave simulation
                speed_generated[i, j, simulation_i] = np.random.randint(sensor_waves.shape[1])
                direction_ind = np.random.randint(sensor_waves.shape[0])

                # overall direction calculated in the generate_sensor_waves script
                # direction for wave simulation
                generate_direction[simulation_i, :] = direction[direction_ind, speed_generated[i, j, simulation_i], :]
                generate_direction_smooth[simulation_i, :] = (
                    direction_smooth[direction_ind, speed_generated[i, j, simulation_i], :]
                )
                generate_direction_pca[simulation_i, :] = (
                    direction_pca[direction_ind, speed_generated[i, j, simulation_i], :]
                )

                # generate brain noise based on dense matrix
                brain_noise = generate_brain_noise(G=G_dense)
                brain_noise_norm[:, :, simulation_i] = (
                        brain_noise[:, : sensor_waves.shape[3]]
                        / np.linalg.norm(brain_noise[:, : sensor_waves.shape[3]])
                )  # normalized

                wave_picked = (
                    sensor_waves[direction_ind, speed_generated[i, j, simulation_i], :, :]
                )

                # normalized wave
                wave_picked_norm = wave_picked / np.linalg.norm(wave_picked)

                # wave + noise
                data = snr_list[j] * wave_picked_norm + brain_noise_norm[:, :, simulation_i]

                # Generate basis waves using sparse cortical model starting from the initially picked point
                # (with spatial error)
                [
                    sensor_waves,
                    _,
                    _,
                    direction,
                    direction_smooth,
                    direction_pca,
                ] = create_waves_on_sensors(
                    cortex=cortex,
                    cortex_smooth=cortex_smooth,
                    params=wave_generation_params,
                    G=G,
                    start_point=starting_source_sparse[simulation_i],
                    spherical=False
                )

                # Solve the LASSO problem without intercept
                [
                    score_fit[simulation_i],
                    _,
                    best_coefs,
                    _,
                    speed_estimated[i, j, simulation_i],
                ] = LASSO_inverse_solve(data=data, waves=sensor_waves, fit_intercept=False)

                direction_ind_estimated = np.argmax(best_coefs)

                # error in direction predicted (out of 1)
                direction_error[i, j, sim_n] = 1 - abs(
                    direction[direction_ind_estimated, speed_estimated[i, j, simulation_i], :]
                    @ generate_direction[simulation_i, :]
                    / np.linalg.norm(
                        direction[
                            direction_ind_estimated, speed_estimated[i, j, simulation_i], :
                        ]
                    )
                    / np.linalg.norm(generate_direction[simulation_i, :])
                )

                direction_error_smooth[i, j, sim_n] = 1 - abs(
                    direction_smooth[
                        direction_ind_estimated, speed_estimated[i, j, sim_n], :
                    ]
                    @ generate_direction_smooth[sim_n, :]
                    / np.linalg.norm(
                        direction_smooth[
                            direction_ind_estimated, speed_estimated[i, j, sim_n], :
                        ]
                    )
                    / np.linalg.norm(generate_direction_smooth[sim_n, :])
                )

                direction_error_pca[i, j, sim_n] = 1 - abs(
                    direction_pca[
                        direction_ind_estimated, speed_estimated[i, j, sim_n], :
                    ]
                    @ generate_direction_pca[sim_n, :]
                    / np.linalg.norm(
                        direction_pca[
                            np.argmax(best_coefs), speed_estimated[i, j, sim_n], :
                        ]
                    )
                    / np.linalg.norm(generate_direction_pca[sim_n, :])
                )

                print(i, j, sim_n)

            # next simulation_number trials without waves, only with static oscillating blobs
            for sim_n in range(simulation_number, 2 * simulation_number):
                idx_dense = src_idx_dense[sim_n - simulation_number]
                idx = src_idx[sim_n - simulation_number]
                [sensor_blob, _] = create_blob_on_sensors(
                    cortex=cortex_dense,
                    params=wave_generation_params,
                    G=G_dense,
                    start_point=idx_dense,
                    T=T
                )

                [sensor_waves, _] = create_waves_on_sensors(
                    cortex=cortex,
                    cortex_smooth=cortex_smooth,
                    params=wave_generation_params,
                    G=G,
                    start_point=idx,
                    spherical=False
                )

                brain_noise = brain_noise_norm[:, :, sim_n - simulation_number]
                sensor_blob_norm = sensor_blob / np.linalg.norm(sensor_blob)
                data = snr_list[i] * sensor_blob_norm + brain_noise

                [score_fit[sim_n], _] = LASSO_inverse_solve(data=data, waves=sensor_waves, fit_intercept=False)
                print(sim_n)

            y_score = score_fit
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
            auc[i, j] = metrics.roc_auc_score(y_true, y_score)
            plt.plot(
                fpr,
                tpr,
                lw=2,
                label="ROC curve for SNR {0}, (area = {1:0.2f})".format(
                    snr_list[j], auc[i, j]
                ),
            )

        plt.title(
            "Mean spatial jitter = "
            + str((spatial_jitter_list[i] + 0.003 / 2) * 1000)
            + " mm"
        )
        plt.legend(loc="lower right")
        plt.show()

    plt.figure()
    plt.subplot(1, 3, 1)
    for j in range(0, spatial_jitter_num):
        plt.plot(snr_list, np.mean(direction_error[j, :, :], axis=1), "o-")
        plt.title("Direction detection error (with curvature)")
        plt.ylim([0, 1])
        plt.xlabel("SNR")
        plt.ylabel("1 - correlation between generated and detected")
    plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)
    plt.subplot(1, 3, 2)
    for j in range(0, spatial_jitter_num):
        plt.plot(snr_list, np.mean(direction_error_smooth[j, :, :], axis=1), "o-")
        plt.title("Direction detection error (on smooth cortex)")
        plt.ylim([0, 1])
        plt.xlabel("SNR")
        plt.ylabel("1 - correlation between generated and detected")
    plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)
    plt.subplot(1, 3, 3)
    for j in range(0, spatial_jitter_num):
        plt.plot(snr_list, np.mean(direction_error_pca[j, :, :], axis=1), "o-")
        plt.title("Direction detection error (main direction)")
        plt.xlabel("SNR")
        plt.ylim([0, 1])
        plt.ylabel("Error between detected and generated speeds in m/s")
    plt.legend((np.array(spatial_jitter_list) + 0.0015) * 1000)

    plt.figure()
    k = 1
    for j in range(0, spatial_jitter_num):
        for s in range(0, speed_generated.shape[0]):
            jitter_1 = 0.2 * np.random.rand(speed_estimated.shape[2])
            jitter_2 = 0.2 * np.random.rand(speed_estimated.shape[2])
            plt.subplot(spatial_jitter_num, snr_num, k)
            plt.scatter(
                speed_generated[j, s, :] + jitter_1, speed_estimated[j, s, :] + jitter_2
            )
            plt.xlabel("True speed, m/s")
            plt.ylabel("Estimated speed, m/s")
            plt.plot(np.arange(-1, 10, 0.1), np.arange(-1, 10, 0.1), "--")
            plt.xlim([-1, 10])
            plt.ylim([-1, 10])
            plt.xticks(range(0, 10), wave_generation_params["speeds"])  # Set locations and labels
            plt.yticks(range(0, 10), wave_generation_params["speeds"])  # Set locations and labels
            plt.title(
                "SNR level = "
                + str(snr_list[s])
                + ", Mean spatial jitter = "
                + str((spatial_jitter_list[j] + 0.0015) * 1000)
                + " mm"
            )
            k += 1

    return (
        auc,
        direction_error,
        direction_error_smooth,
        direction_error_pca,
        speed_generated,
        speed_estimated,
    )
