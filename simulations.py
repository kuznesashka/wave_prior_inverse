from create_blob_on_sensors import create_blob_on_sensors
from create_waves_on_sensors import create_waves_on_sensors
from generate_brain_noise import generate_brain_noise
from LASSO_inverse_solve import LASSO_inverse_solve
import scipy.io
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def simulations(data_dir, channel_type, params, snr, num_sim=100):
    """Function to run Monte Carlo simulations
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

    # uploading sparse and dense forward model and cortical model
    G_raw = scipy.io.loadmat(data_dir+'/G.mat')
    cortex_raw = scipy.io.loadmat(data_dir+'/cortex.mat')
    G_dense_raw = scipy.io.loadmat(data_dir+'/G_medium.mat')
    cortex_dense_raw = scipy.io.loadmat(data_dir+'/cortex_medium.mat')

    # pick the appropriate channels
    if channel_type == 'mag':
        G = G_raw['G'][np.arange(2, 306, 3)]  # magnetometers
        G_dense = G_dense_raw['G'][np.arange(2, 306, 3)]  # magnetometers
    elif channel_type == 'grad':
        G = G_raw['G'][np.setdiff1d(range(0, 306), np.arange(2, 306, 3))]  # gradiometers
        G_dense = G_dense_raw['G'][np.setdiff1d(range(0, 306), np.arange(2, 306, 3))]  # gradiometers
    else:
        print('Wrong channel name')

    cortex = cortex_raw['cortex'][0]
    cortex_dense = cortex_dense_raw['cortex'][0]
    vertices = cortex[0][1]
    vertices_dense = cortex_dense[0][1]

    speeds = params['speeds']  # speed range
    T = int(params['duration']*params['Fs']+1)*2  # duration in time

    y_true = np.zeros(num_sim*2)  # true labels
    y_true[0:num_sim] = np.ones(num_sim)
    auc = np.zeros(len(snr))

    # ROC figure
    plt.figure()
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    speed_error = np.zeros([len(snr), num_sim])  # error in speed detection
    direction_error = np.zeros([len(snr), num_sim])  # error in direction detection

    k = 0
    for snr_level in snr:
        score_fit = np.zeros(2*num_sim)  # R-squared metrics for all simulations
        generate_direction = np.zeros([num_sim, 3])  # true directions modeled
        generate_speed = np.zeros(num_sim, dtype=int)  # speed modeled
        src_idx = np.zeros(num_sim, dtype=int)  # assumed starting source from the sparse model
        src_idx_dense = np.zeros(num_sim, dtype=int)  # true starting source from the dense model
        brain_noise_norm = np.zeros([G.shape[0], T, num_sim])  # brain noise array
        best_intercept = np.zeros(2*num_sim)  # best intercepts for LASSO with intercept

        # in first num_sim trials the data is generated using traveling waves
        for sim_n in range(0, num_sim):
            src_idx[sim_n] = np.random.randint(0, G.shape[1])  # random starting source from sparse cortical model
            # compute distance from starting source to all sources from dense cortical model
            dist = np.sum(np.sqrt((np.repeat(vertices[src_idx[sim_n], np.newaxis],
                                             vertices_dense.shape[0], axis=0) - vertices_dense)**2), axis=1)
            # find close sources
            ind_close = np.where((dist > 0.002)&(dist <= 0.005))[0]
            src_idx_dense[sim_n] = ind_close[np.random.randint(0, len(ind_close))]  # pick randomly new starting source
            [sensor_waves, direction, path_final] = create_waves_on_sensors(cortex_dense, params,
                                                                            G_dense, src_idx_dense[sim_n], spherical=False)
            generate_speed[sim_n] = np.random.randint(0, sensor_waves.shape[1])  # speed for wave simulation
            direction_ind = np.random.randint(0, sensor_waves.shape[0])
            generate_direction[sim_n, :] = direction[direction_ind, generate_speed[sim_n], :]  # direction for wave simulation

            brain_noise = generate_brain_noise(G_dense)  # generate brain noise based on dense matrix
            brain_noise_norm[:, :, sim_n] = brain_noise[:, :sensor_waves.shape[3]]/\
                                            np.linalg.norm(brain_noise[:, :sensor_waves.shape[3]])  # normalized
            wave_picked = sensor_waves[direction_ind, generate_speed[sim_n], :, :]
            wave_picked_norm = wave_picked/np.linalg.norm(wave_picked)  # normalized wave
            data = snr_level*wave_picked_norm + brain_noise_norm[:, :, sim_n]  # wave + noise

            # visualization
            # plt.figure()
            # plt.plot(np.concatenate((brain_noise_norm[:, :, sim_n], wave_picked_norm), axis=1).T)
            # plt.figure()
            # plt.plot(data.T)

            # Generate basis waves using sparse cortical model starting from the initially picked point
            # (with spatial error)
            [sensor_waves, direction, path_final] = create_waves_on_sensors(cortex, params, G,
                                                                               src_idx[sim_n], spherical=False)
            # Solve the LASSO problem without intercept
            [score_fit[sim_n], best_intercept[sim_n], best_coefs, best_shift, best_speed_ind] = \
                LASSO_inverse_solve(data, sensor_waves, False)
            # error in speed predicted (m/s)
            speed_error[k, sim_n] = np.abs(speeds[best_speed_ind] - speeds[generate_speed[sim_n]])
            # error in direction predicted (out of 1)
            direction_error[k, sim_n] = 1 - direction[np.argmax(best_coefs), best_speed_ind, :] @ \
                                        generate_direction[sim_n, :] / \
                                        np.linalg.norm(direction[np.argmax(best_coefs), best_speed_ind, :]) / \
                                        np.linalg.norm(generate_direction[sim_n, :])
            print(sim_n)

        # next num_sim trials without waves, only with static oscillating blobs
        for sim_n in range(num_sim, 2*num_sim):
            idx_dense = src_idx_dense[sim_n - num_sim]
            idx = src_idx[sim_n - num_sim]
            [sensor_blob, path_indices] = create_blob_on_sensors(cortex_dense, params, G_dense, idx_dense, T)
            [sensor_waves, direction, path_final] = create_waves_on_sensors(cortex, params, G, idx, spherical=False)

            brain_noise = brain_noise_norm[:, :, sim_n-num_sim]
            sensor_blob_norm = sensor_blob/np.linalg.norm(sensor_blob)
            data = snr_level*sensor_blob_norm + brain_noise

            # plt.figure()
            # plt.plot(data.T)

            [score_fit[sim_n], best_intercept[sim_n], best_coefs, best_shift, best_speed_ind] = \
                LASSO_inverse_solve(data, sensor_waves, False)
            print(sim_n)

        y_score = score_fit
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc[k] = metrics.roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, lw=lw, label='ROC curve for SNR {0}, (area = {1:0.2f})'.format(snr_level, auc[k]))
        k += 1

    plt.title('Receiver operating characteristics for different SNR')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(snr, np.mean(direction_error, axis=1), 'o-')
    plt.title('Direction detection error')
    plt.ylim([0, 1])
    plt.xlabel('SNR')
    plt.ylabel('1 - correlation between generated and detected')
    plt.subplot(2, 1, 2)
    plt.plot(snr, np.mean(speed_error, axis=1), 'o-')
    plt.title('Speed detection error')
    plt.xlabel('SNR')
    plt.ylabel('Error between detected and generated speeds in m/s')

    return auc, speed_error, direction_error

