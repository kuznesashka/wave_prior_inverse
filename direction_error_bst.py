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
    params: Dict[str, Any],
    snr: List[float],
    spatial_jitter: List[float],
    num_sim: int = 100,
):
    """Function to run Monte Carlo simulations
    Focus on speed and direction error: as a function of spatial jitter, snr
    All models and forward operators are generated in brainstorm
        Parameters
        ----------
        data_dir : str
            Data directory with G.mat and cortex.mat
        channel_type : str
            Type of channels used 'grad' or 'mag'
        params : Dict[str, Any]
            Wave modeling parameters
        snr : List[float]
            List with all considered snr values
        spatial_jitter : List[float]
            List of lower bounds of spatial jitter values (in m) with spatial jitter distributed uniformly
            between [jitter[i], jitter[i]+0.003]
        num_sim : int
            Number of simulations for one class
        Returns
        -------
        auc : AUC values for all snr levels
        speed_error : difference between simulated and detected speeds in m/s
        direction_error : 1-correlation between generated and detected directions
        ROC curve plot, Error plots
    """

    # 1. UPLOADING FORWARD OPERATOR AND CORTICAL MODEL FROM BST
    # uploading sparse and dense forward model and cortical model
    G_dense_raw = scipy.io.loadmat(data_dir + "/G_low.mat")
    cortex_dense_raw = scipy.io.loadmat(data_dir + "/cortex_low.mat")
    cortex_smooth_dense_raw = scipy.io.loadmat(data_dir + "/cortex_smooth_low.mat")
    G_raw = scipy.io.loadmat(data_dir + "/G.mat")
    cortex_raw = scipy.io.loadmat(data_dir + "/cortex.mat")
    cortex_smooth_raw = scipy.io.loadmat(data_dir + "/cortex_smooth.mat")

    # pick the appropriate channels
    if channel_type == "mag":
        G_dense = G_dense_raw["G"][np.arange(2, 306, 3)]  # magnetometers
        G = G_raw["G"][np.arange(2, 306, 3)]  # magnetometers
    elif channel_type == "grad":
        G_dense = G_dense_raw["G"][
            np.setdiff1d(range(0, 306), np.arange(2, 306, 3))
        ]  # gradiometers
        G = G_raw["G"][
            np.setdiff1d(range(0, 306), np.arange(2, 306, 3))
        ]  # magnetometers
    else:
        print("Wrong channel name")

    cortex_dense = cortex_dense_raw["cortex"][0]
    cortex = cortex_raw["cortex"][0]
    cortex_smooth = cortex_smooth_raw["cortex_smooth"][0]
    cortex_smooth_dense = cortex_smooth_dense_raw["cortex_smooth"][0]

    vertices = cortex[0][1]
    vertices_dense = cortex_dense[0][1]
    vertices_dense_smooth = cortex_smooth_dense[0][0]
    assert vertices_dense.shape == vertices_dense_smooth.shape
    faces_dense = cortex_dense[0][0] - 1
    assert faces_dense.shape[1] == 3
    assert faces_dense.shape[0] > vertices_dense.shape[0]

    # 2. PARAMETERS
    T = int(params["duration"] * params["Fs"] + 1) * 2  # duration in time

    speed_generated = np.zeros([len(spatial_jitter), len(snr), num_sim], dtype=int)
    speed_estimated = np.zeros([len(spatial_jitter), len(snr), num_sim], dtype=int)
    direction_error = np.zeros([len(spatial_jitter), len(snr), num_sim])
    direction_error_smooth = np.zeros([len(spatial_jitter), len(snr), num_sim])
    direction_error_pca = np.zeros([len(spatial_jitter), len(snr), num_sim])
    spatial_error = np.zeros([len(spatial_jitter), len(snr), num_sim])

    # 3. ROC CURVE
    y_true = np.zeros(num_sim * 2)  # true labels
    y_true[0:num_sim] = np.ones(num_sim)
    auc = np.zeros([len(snr), len(spatial_jitter)])

    # ROC figure
    plt.figure()

    for i in range(0, len(spatial_jitter)):
        plt.subplot(1, len(spatial_jitter), (i + 1))
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        for j in range(0, len(snr)):
            score_fit = np.zeros(2 * num_sim)  # R-squared metrics for all simulations

            # true directions modeled
            generate_direction = np.zeros([num_sim, 3])
            generate_direction_smooth = np.zeros([num_sim, 3])
            generate_direction_pca = np.zeros([num_sim, 3])

            # assumed starting source from the sparse model
            src_idx = np.zeros(num_sim, dtype=int)

            # true starting source from the dense model
            src_idx_dense = np.zeros(num_sim, dtype=int)

            # brain noise array
            brain_noise_norm = np.zeros([G_dense.shape[0], T, num_sim])

            # in first num_sim trials the data is generated using traveling waves
            for sim_n in range(0, num_sim):
                # random starting source from sparse cortical model
                src_idx[sim_n] = np.random.randint(0, G.shape[1])

                # compute distance from starting source to all sources from dense cortical model
                dist = np.sum(
                    np.sqrt(
                        (
                            np.repeat(
                                vertices[src_idx[sim_n], np.newaxis],
                                vertices_dense.shape[0],
                                axis=0,
                            )
                            - vertices_dense
                        )
                        ** 2
                    ),
                    axis=1,
                )

                # find close sources
                ind_close = np.where(
                    (dist > spatial_jitter[i]) & (dist <= (spatial_jitter[i] + 0.003))
                )[0]
                if len(ind_close) == 0:
                    continue

                # pick randomly new starting source
                src_idx_dense[sim_n] = ind_close[np.random.randint(0, len(ind_close))]
                spatial_error[i, j, sim_n] = np.linalg.norm(
                    vertices_dense[src_idx_dense[sim_n]] - vertices[src_idx[sim_n]]
                )

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
                    params=params,
                    G=G_dense,
                    start_point=src_idx_dense[sim_n],
                    spherical=False
                )

                # speed for wave simulation
                speed_generated[i, j, sim_n] = np.random.randint(0, sensor_waves.shape[1])
                direction_ind = np.random.randint(0, sensor_waves.shape[0])

                # overall direction calculated in the generate_sensor_waves script
                # direction for wave simulation
                generate_direction[sim_n, :] = direction[
                    direction_ind, speed_generated[i, j, sim_n], :
                ]
                generate_direction_smooth[sim_n, :] = direction_smooth[
                    direction_ind, speed_generated[i, j, sim_n], :
                ]
                generate_direction_pca[sim_n, :] = direction_pca[
                    direction_ind, speed_generated[i, j, sim_n], :
                ]

                # generate brain noise based on dense matrix
                brain_noise = generate_brain_noise(G=G_dense)
                brain_noise_norm[:, :, sim_n] = (
                        brain_noise[:, : sensor_waves.shape[3]]
                        / np.linalg.norm(brain_noise[:, : sensor_waves.shape[3]])
                )  # normalized

                wave_picked = sensor_waves[
                    direction_ind, speed_generated[i, j, sim_n], :, :
                ]

                # normalized wave
                wave_picked_norm = wave_picked / np.linalg.norm(wave_picked)

                # wave + noise
                data = snr[j] * wave_picked_norm + brain_noise_norm[:, :, sim_n]

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
                    params=params,
                    G=G,
                    start_point=src_idx[sim_n],
                    spherical=False
                )

                # Solve the LASSO problem without intercept
                [
                    score_fit[sim_n],
                    _,
                    best_coefs,
                    _,
                    speed_estimated[i, j, sim_n],
                ] = LASSO_inverse_solve(data=data, waves=sensor_waves, fit_intercept=False)

                direction_ind_estimated = np.argmax(best_coefs)

                # error in direction predicted (out of 1)
                direction_error[i, j, sim_n] = 1 - abs(
                    direction[direction_ind_estimated, speed_estimated[i, j, sim_n], :]
                    @ generate_direction[sim_n, :]
                    / np.linalg.norm(
                        direction[
                            direction_ind_estimated, speed_estimated[i, j, sim_n], :
                        ]
                    )
                    / np.linalg.norm(generate_direction[sim_n, :])
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

            # next num_sim trials without waves, only with static oscillating blobs
            for sim_n in range(num_sim, 2 * num_sim):
                idx_dense = src_idx_dense[sim_n - num_sim]
                idx = src_idx[sim_n - num_sim]
                [sensor_blob, _] = create_blob_on_sensors(
                    cortex=cortex_dense,
                    params=params,
                    G=G_dense,
                    start_point=idx_dense,
                    T=T
                )

                [sensor_waves, _] = create_waves_on_sensors(
                    cortex=cortex,
                    cortex_smooth=cortex_smooth,
                    params=params,
                    G=G,
                    start_point=idx,
                    spherical=False
                )

                brain_noise = brain_noise_norm[:, :, sim_n - num_sim]
                sensor_blob_norm = sensor_blob / np.linalg.norm(sensor_blob)
                data = snr[i] * sensor_blob_norm + brain_noise

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
                    snr[j], auc[i, j]
                ),
            )

        plt.title(
            "Mean spatial jitter = "
            + str((spatial_jitter[i] + 0.003 / 2) * 1000)
            + " mm"
        )
        plt.legend(loc="lower right")
        plt.show()

    plt.figure()
    plt.subplot(1, 3, 1)
    for j in range(0, len(spatial_jitter)):
        plt.plot(snr, np.mean(direction_error[j, :, :], axis=1), "o-")
        plt.title("Direction detection error (with curvature)")
        plt.ylim([0, 1])
        plt.xlabel("SNR")
        plt.ylabel("1 - correlation between generated and detected")
    plt.legend((np.array(spatial_jitter) + 0.0015) * 1000)
    plt.subplot(1, 3, 2)
    for j in range(0, len(spatial_jitter)):
        plt.plot(snr, np.mean(direction_error_smooth[j, :, :], axis=1), "o-")
        plt.title("Direction detection error (on smooth cortex)")
        plt.ylim([0, 1])
        plt.xlabel("SNR")
        plt.ylabel("1 - correlation between generated and detected")
    plt.legend((np.array(spatial_jitter) + 0.0015) * 1000)
    plt.subplot(1, 3, 3)
    for j in range(0, len(spatial_jitter)):
        plt.plot(snr, np.mean(direction_error_pca[j, :, :], axis=1), "o-")
        plt.title("Direction detection error (main direction)")
        plt.xlabel("SNR")
        plt.ylim([0, 1])
        plt.ylabel("Error between detected and generated speeds in m/s")
    plt.legend((np.array(spatial_jitter) + 0.0015) * 1000)

    plt.figure()
    k = 1
    for j in range(0, len(spatial_jitter)):
        for s in range(0, speed_generated.shape[0]):
            jitter_1 = 0.2 * np.random.rand(speed_estimated.shape[2])
            jitter_2 = 0.2 * np.random.rand(speed_estimated.shape[2])
            plt.subplot(len(spatial_jitter), len(snr), k)
            plt.scatter(
                speed_generated[j, s, :] + jitter_1, speed_estimated[j, s, :] + jitter_2
            )
            plt.xlabel("True speed, m/s")
            plt.ylabel("Estimated speed, m/s")
            plt.plot(np.arange(-1, 10, 0.1), np.arange(-1, 10, 0.1), "--")
            plt.xlim([-1, 10])
            plt.ylim([-1, 10])
            plt.xticks(range(0, 10), params["speeds"])  # Set locations and labels
            plt.yticks(range(0, 10), params["speeds"])  # Set locations and labels
            plt.title(
                "SNR level = "
                + str(snr[s])
                + ", Mean spatial jitter = "
                + str((spatial_jitter[j] + 0.0015) * 1000)
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
