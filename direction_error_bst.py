from create_waves_on_sensors import create_waves_on_sensors
from generate_brain_noise import generate_brain_noise
from LASSO_inverse_solve import LASSO_inverse_solve
import scipy.io
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from mayavi.mlab import *


def direction_error_bst(data_dir, channel_type, params, snr, spatial_jitter, num_sim=100):
    """Function to run Monte Carlo simulations
        Focus on speed and direction error: as a function of spatial jitter, snr
        All models and forward operators are generated in brainstorm
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
            spatial_jitter: list
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
    G_dense_raw = scipy.io.loadmat(data_dir + '/G_medium.mat')
    cortex_dense_raw = scipy.io.loadmat(data_dir + '/cortex_medium.mat')
    cortex_smooth_dense_raw = scipy.io.loadmat(data_dir + '/cortex_smooth_medium.mat')
    G_raw = scipy.io.loadmat(data_dir + '/G.mat')
    cortex_raw = scipy.io.loadmat(data_dir + '/cortex.mat')

    # pick the appropriate channels
    if channel_type == 'mag':
        G_dense = G_dense_raw['G'][np.arange(2, 306, 3)]  # magnetometers
        G = G_raw['G'][np.arange(2, 306, 3)]  # magnetometers
    elif channel_type == 'grad':
        G_dense = G_dense_raw['G'][np.setdiff1d(range(0, 306), np.arange(2, 306, 3))]  # gradiometers
        G = G_raw['G'][np.setdiff1d(range(0, 306), np.arange(2, 306, 3))]  # magnetometers
    else:
        print('Wrong channel name')

    cortex_dense = cortex_dense_raw['cortex'][0]
    cortex_smooth_dense = cortex_smooth_dense_raw['cortex'][0]
    vertices_dense = cortex_dense[0][1]

    # Visualization of 3d brain surfaces
    # triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, color=(0, 1, 1))

    # 2. PARAMETERS
    # TODO: fix duration
    speeds = params['speeds']  # speed range
    T = int(params['duration']*params['Fs']+1)*2  # duration in time

    speed_generated = np.zeros([len(snr), len(spatial_jitter), num_sim], dtype=int)
    speed_estimated = np.zeros([len(snr), len(spatial_jitter), num_sim], dtype=int)
    direction_error = np.zeros([len(snr), len(spatial_jitter), num_sim])
    direction_error_smooth = np.zeros([len(snr), len(spatial_jitter), num_sim])
    direction_error_pca = np.zeros([len(snr), len(spatial_jitter), num_sim])

    # direction_error = np.zeros([len(snr), num_sim])  # error in direction detection

    # 3. ROC CURVE
    # y_true = np.zeros(num_sim * 2)  # true labels
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

    for i in range(0, len(snr)):
        for j in range(0, len(spatial_jitter)):
            score_fit = np.zeros(2*num_sim)  # R-squared metrics for all simulations
            generate_direction = np.zeros([num_sim, 3])  # true directions modeled
            generate_direction_smooth = np.zeros([num_sim, 3])  # true directions modeled
            generate_direction_pca = np.zeros([num_sim, 3])  # true directions modeled
            # src_idx = np.zeros(num_sim, dtype=int)  # assumed starting source from the sparse model
            src_idx_dense = np.zeros(num_sim, dtype=int)  # true starting source from the dense model
            brain_noise_norm = np.zeros([G_dense.shape[0], T, num_sim])  # brain noise array

            # in first num_sim trials the data is generated using traveling waves
            for sim_n in range(0, num_sim):
                src_idx_dense[sim_n] = np.random.randint(0, G_dense.shape[1])  # random starting source from sparse cortical model
                # compute distance from starting source to all sources from dense cortical model
                # dist = np.sum(np.sqrt((np.repeat(vertices[src_idx[sim_n], np.newaxis],
                #                                  vertices_dense.shape[0], axis=0) - vertices_dense)**2), axis=1)
                # # find close sources
                # ind_close = np.where((dist > 0.002)&(dist <= 0.005))[0]
                # src_idx_dense[sim_n] = ind_close[np.random.randint(0, len(ind_close))]  # pick randomly new starting source
                [sensor_waves, direction, direction_smooth, direction_pca] = create_waves_on_sensors(cortex_dense, cortex_smooth_dense, params,
                                                                            G_dense, src_idx_dense[sim_n], spherical=False)
                speed_generated[i, j, sim_n] = np.random.randint(0, sensor_waves.shape[1])  # speed for wave simulation
                direction_ind = np.random.randint(0, sensor_waves.shape[0])
                generate_direction[sim_n, :] = direction[direction_ind, speed_generated[i, j, sim_n], :]  # direction for wave simulation
                generate_direction_smooth[sim_n, :] = direction_smooth[direction_ind, speed_generated[i, j, sim_n], :]  # direction for wave simulation
                generate_direction_pca[sim_n, :] = direction_pca[direction_ind, speed_generated[i, j, sim_n], :]  # direction for wave simulation

                brain_noise = generate_brain_noise(G_dense)  # generate brain noise based on dense matrix
                brain_noise_norm[:, :, sim_n] = brain_noise[:, :sensor_waves.shape[3]]/\
                                            np.linalg.norm(brain_noise[:, :sensor_waves.shape[3]])  # normalized
                wave_picked = sensor_waves[direction_ind, speed_generated[i, j, sim_n], :, :]
                wave_picked_norm = wave_picked/np.linalg.norm(wave_picked)  # normalized wave
                data = snr[i]*wave_picked_norm + brain_noise_norm[:, :, sim_n]  # wave + noise

                # plt.figure()
                # plt.plot(np.concatenate((brain_noise_norm[:, :, sim_n], wave_picked_norm), axis=1).T)
                # plt.figure()
                # plt.plot(data.T)

                # Generate basis waves using sparse cortical model starting from the initially picked point
                # (with spatial error)
                [sensor_waves, direction, direction_smooth, direction_pca] = create_waves_on_sensors(cortex_dense, cortex_smooth_dense, params, G_dense,
                                                                                   src_idx_dense[sim_n], spherical=False)
                # Solve the LASSO problem without intercept
                [score_fit[sim_n], best_intercept, best_coefs, best_shift, speed_estimated[i, j, sim_n]] = \
                        LASSO_inverse_solve(data, sensor_waves, False)

                # error in direction predicted (out of 1)
                direction_error[i, j, sim_n] = 1 - abs(direction[np.argmax(best_coefs), speed_generated[i, j, sim_n], :] @ \
                                            generate_direction[sim_n, :] / \
                                            np.linalg.norm(direction[np.argmax(best_coefs), speed_estimated[i, j, sim_n], :]) / \
                                            np.linalg.norm(generate_direction[sim_n, :]))
                direction_error_smooth[i, j, sim_n] = 1 - abs(direction_smooth[np.argmax(best_coefs), speed_generated[i, j, sim_n], :] @ \
                                               generate_direction_smooth[sim_n, :] / np.linalg.norm(direction_smooth[np.argmax(best_coefs),
                                                speed_estimated[i, j, sim_n], :]) / np.linalg.norm(generate_direction_smooth[sim_n, :]))
                direction_error_pca[i, j, sim_n] = 1 - abs(direction_pca[np.argmax(best_coefs), speed_generated[i, j, sim_n], :] @ \
                                               generate_direction_pca[sim_n, :] / np.linalg.norm(direction_pca[np.argmax(best_coefs), speed_estimated[i, j, sim_n], :]) / \
                                               np.linalg.norm(generate_direction_pca[sim_n, :]))
                print(i, j, sim_n)

        # next num_sim trials without waves, only with static oscillating blobs
    #     for sim_n in range(num_sim, 2*num_sim):
    #         idx_dense = src_idx_dense[sim_n - num_sim]
    #         idx = src_idx[sim_n - num_sim]
    #         [sensor_blob, path_indices] = create_blob_on_sensors(cortex_dense, params, G_dense, idx_dense, T)
    #         [sensor_waves, direction, path_final] = create_waves_on_sensors(cortex, params, G, idx, spherical=False)
    #
    #         brain_noise = brain_noise_norm[:, :, sim_n-num_sim]
    #         sensor_blob_norm = sensor_blob/np.linalg.norm(sensor_blob)
    #         data = snr_level*sensor_blob_norm + brain_noise
    #
    #         # plt.figure()
    #         # plt.plot(data.T)
    #
    #         [score_fit[sim_n], best_intercept[sim_n], best_coefs, best_shift, best_speed_ind] = \
    #             LASSO_inverse_solve(data, sensor_waves, False)
    #         print(sim_n)
    #
    #     y_score = score_fit
    #     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    #     auc[k] = metrics.roc_auc_score(y_true, y_score)
    #     plt.plot(fpr, tpr, lw=lw, label='ROC curve for SNR {0}, (area = {1:0.2f})'.format(snr_level, auc[k]))
    #     k += 1
    #
    # plt.title('Receiver operating characteristics for different SNR')
    # plt.legend(loc="lower right")
    # plt.show()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(snr, np.mean(direction_error, axis=2), 'o-')
    plt.title('Direction detection error (with curvature)')
    plt.ylim([0, 1])
    plt.xlabel('SNR')
    plt.ylabel('1 - correlation between generated and detected')
    plt.subplot(1, 3, 2)
    plt.plot(snr, np.mean(direction_error_smooth, axis=2), 'o-')
    plt.title('Direction detection error (on smooth cortex)')
    plt.ylim([0, 1])
    plt.xlabel('SNR')
    plt.ylabel('1 - correlation between generated and detected')
    plt.subplot(1, 3, 3)
    plt.plot(snr, np.mean(direction_error_pca, axis=2), 'o-')
    plt.title('Direction detection error (main direction)')
    plt.xlabel('SNR')
    plt.ylim([0, 1])
    plt.ylabel('Error between detected and generated speeds in m/s')


    for j in range(0, len(spatial_jitter)):
        for s in range(0, speed_generated.shape[0]):
            jitter_1 = 0.2 * np.random.rand(speed_estimated.shape[2])
            jitter_2 = 0.2 * np.random.rand(speed_estimated.shape[2])
            plt.subplot(len(spatial_jitter), len(snr), (j + s + 1))
            plt.scatter(speed_generated[s, :] + jitter_1, speed_estimated[s, :] + jitter_2)
            plt.xlabel('True speed, m/s')
            plt.ylabel('Estimated speed, m/s')
            plt.plot(np.arange(-1, 10, 0.1), np.arange(-1, 10, 0.1), '--')
            plt.xlim([-1, 10])
            plt.ylim([-1, 10])
            plt.title(('SNR level, spatial jitter').format((snr[s], spatial_jitter[j])))
            plt.xticks(range(0, 10), params['speeds'])  # Set locations and labels
            plt.yticks(range(0, 10), params['speeds'])  # Set locations and labels

    return
