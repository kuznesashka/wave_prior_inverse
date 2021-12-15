from create_waves_on_sensors import create_waves_on_sensors
from generate_brain_noise import generate_brain_noise
from LASSO_inverse_solve import LASSO_inverse_solve
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def simulations_speed_error(data_dir, channel_type, params, snr, num_sim=100):
    """Function to run Monte Carlo simulations without spatial jitter and control only speed detection error
    Parameters
    ----------
    data_dir : str
        Data directory with G.mat and cortex.mat
    channel_type : str
        Type of channels used 'grad' or 'mag'
    params : dict
        Wave modeling parameters
    snr : list
        List with all considered snr values
    num_sim : int
        Number of simulations for one class
    Returns
    -------
    auc : AUC values for all snr levels
    speed_error : difference between simulated and detected speeds in m/s
    direction_error : 1-correlation between generated and detected directions
    ROC curve plot, Error plots
    """

    # uploading only dense forward model and cortical model
    G_dense_raw = scipy.io.loadmat(data_dir + "/G_medium.mat")
    cortex_dense_raw = scipy.io.loadmat(data_dir + "/cortex_medium.mat")

    # pick the appropriate channels
    if channel_type == "mag":
        G_dense = G_dense_raw["G"][np.arange(2, 306, 3)]  # magnetometers
    elif channel_type == "grad":
        G_dense = G_dense_raw["G"][
            np.setdiff1d(range(0, 306), np.arange(2, 306, 3))
        ]  # gradiometers
    else:
        print("Wrong channel name")

    cortex_dense = cortex_dense_raw["cortex"][0]
    vertices_dense = cortex_dense[0][1]

    speeds = params["speeds"]  # speed range
    T = int(params["duration"] * params["Fs"] + 1) * 2  # duration in time

    # y_true = np.zeros(num_sim*2)  # true labels
    # y_true[0:num_sim] = np.ones(num_sim)
    # auc = np.zeros(len(snr))
    #
    # # ROC figure
    # plt.figure()
    # lw = 2
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')

    k = 0
    speed_generated = np.zeros([len(snr), num_sim], dtype=int)  # speed modeled
    speed_estim = np.zeros([len(snr), num_sim], dtype=int)  # speed estimated

    for snr_level in snr:
        score_fit = np.zeros(2 * num_sim)  # R-squared metrics for all simulations
        generate_direction = np.zeros([num_sim, 3])  # true directions modeled
        src_idx_dense = np.zeros(
            num_sim, dtype=int
        )  # true starting source from the dense model
        brain_noise_norm = np.zeros([G_dense.shape[0], T, num_sim])  # brain noise array

        for sim_n in range(0, num_sim):
            src_idx_dense[sim_n] = np.random.randint(
                0, G_dense.shape[1]
            )  # random starting source from dense cortical model

            [sensor_waves, direction, path_final] = create_waves_on_sensors(
                cortex_dense, params, G_dense, src_idx_dense[sim_n], spherical=False
            )
            speed_generated[k, sim_n] = np.random.randint(
                0, sensor_waves.shape[1]
            )  # speed for wave simulation
            direction_ind = np.random.randint(0, sensor_waves.shape[0])

            brain_noise = generate_brain_noise(
                G_dense
            )  # generate brain noise based on dense matrix
            brain_noise_norm[:, :, sim_n] = brain_noise[
                :, : sensor_waves.shape[3]
            ] / np.linalg.norm(
                brain_noise[:, : sensor_waves.shape[3]]
            )  # normalized
            wave_picked = sensor_waves[direction_ind, speed_generated[k, sim_n], :, :]
            wave_picked_norm = wave_picked / np.linalg.norm(
                wave_picked
            )  # normalized wave
            data = (
                snr_level * wave_picked_norm + brain_noise_norm[:, :, sim_n]
            )  # wave + noise

            # plt.figure()
            # plt.plot(np.concatenate((brain_noise_norm[:, :, sim_n], wave_picked_norm), axis=1).T)
            # plt.figure()
            # plt.plot(data.T)

            # Generate basis waves using sparse cortical model starting from the initially picked point
            # (with spatial error)
            [sensor_waves, direction, path_final] = create_waves_on_sensors(
                cortex_dense, params, G_dense, src_idx_dense[sim_n], spherical=False
            )
            # Solve the LASSO problem without intercept
            [
                score_fit[sim_n],
                best_intercept,
                best_coefs,
                best_shift,
                speed_estim[k, sim_n],
            ] = LASSO_inverse_solve(data, sensor_waves, False)
            print(sim_n)

        k += 1

    jitter = 0.1 * np.random.rand(speed_estim.shape[1])

    plt.scatter(speed_generated + jitter, speed_estim + jitter)
    plt.xlabel("True speed index")
    plt.ylabel("Estimated speed index")
    plt.plot(np.arange(0, 9, 0.1), np.arange(0, 9, 0.1), "--")

    return
