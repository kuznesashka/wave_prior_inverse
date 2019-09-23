from create_waves_on_sensors_py import create_waves_on_sensors_py
from generate_brain_noise import generate_brain_noise
from LASSO_inverse_solve import LASSO_inverse_solve
import numpy as np
import matplotlib.pyplot as plt
import mne


def simulations_direction_error(channel_type, params, snr, num_sim=100):
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

    # uploading dense forward model and cortical model

    fwd_dense = mne.read_forward_solution('B1C2_fwd_dense')
    fwd_fixed = mne.convert_forward_solution(fwd_dense, surf_ori=True, force_fixed=True,
                                             use_cps=True)
    G = fwd_fixed['sol']['data']

    # pick the appropriate channels
    if channel_type == 'mag':
        G = G[np.arange(2, 306, 3), :]  # magnetometers
    elif channel_type == 'grad':
        G = G[np.setdiff1d(range(0, 306), np.arange(2, 306, 3)), :]  # gradiometers
    else:
        print('Wrong channel name')

    # vertices_r = fwd['src'][0]['rr']
    # vertices_l = fwd['src'][1]['rr']

    T = int(params['duration']*params['Fs']+1)*2  # duration in time

    k = 0
    speed_generated = np.zeros([len(snr), num_sim], dtype=int)  # speed modeled
    speed_estimated = np.zeros([len(snr), num_sim], dtype=int)  # speed estimated
    direction_error_curve = np.zeros([len(snr), num_sim])  # error in direction detection
    direction_error_sphere = np.zeros([len(snr), num_sim])  # error in direction detection
    direction_error_pca = np.zeros([len(snr), num_sim])  # error in direction detection

    for snr_level in snr:
        score_fit = np.zeros(num_sim)  # R-squared metrics for all simulations
        direction_generated = np.zeros([num_sim, 3])  # true directions modeled
        src_idx_dense = np.zeros(num_sim, dtype=int)  # true starting source from the dense model
        brain_noise_norm = np.zeros([G.shape[0], T, num_sim])  # brain noise array

        for sim_n in range(0, num_sim):
            src_idx_dense[sim_n] = np.random.randint(0, G.shape[1])  # random starting source from sparse cortical model
            [sensor_waves, direction_curved, direction_pca, direction_on_sphere, projected_path] = \
                create_waves_on_sensors_py(fwd_fixed, params, channel_type, src_idx_dense[sim_n], spherical=False)
            speed_generated[k, sim_n] = np.random.randint(0, sensor_waves.shape[1])  # speed for wave simulation
            direction_ind = np.random.randint(0, sensor_waves.shape[0])

            direction_generated_curve = direction_curved[direction_ind, speed_generated[k, sim_n], :]  # direction for wave simulation
            direction_generated_pca = direction_pca[direction_ind, speed_generated[k, sim_n], :]
            direction_generated_sphere = direction_on_sphere[direction_ind, speed_generated[k, sim_n], :]
            projected_path_generate = projected_path[direction_ind, speed_generated[k, sim_n], :, :]

            brain_noise = generate_brain_noise(G)  # generate brain noise based on dense matrix
            brain_noise_norm[:, :, sim_n] = brain_noise[:, :sensor_waves.shape[3]]/\
                                            np.linalg.norm(brain_noise[:, :sensor_waves.shape[3]])  # normalized
            wave_picked = sensor_waves[direction_ind, speed_generated[k, sim_n], :, :]
            wave_picked_norm = wave_picked/np.linalg.norm(wave_picked)  # normalized wave
            data = snr_level*wave_picked_norm + brain_noise_norm[:, :, sim_n]  # wave + noise

            # plt.figure()
            # plt.plot(np.concatenate((brain_noise_norm[:, :, sim_n], wave_picked_norm), axis=1).T)
            # plt.figure()
            # plt.plot(data.T)

            # Generate basis waves using sparse cortical model starting from the initially picked point
            # (with spatial error)
            [sensor_waves, direction_curved, direction_pca, direction_on_sphere, projected_path] = \
                create_waves_on_sensors_py(fwd_fixed, params, channel_type, src_idx_dense[sim_n], spherical=False)
            # Solve the LASSO problem without intercept
            [score_fit[sim_n], best_intercept, best_coefs, best_shift, speed_estimated[k, sim_n]] = \
                LASSO_inverse_solve(data, sensor_waves, False)
            # error in speed predicted (m/s)
            # error in direction predicted (out of 1)
            direction_error_curve[k, sim_n] = 1 - abs(direction_curved[np.argmax(best_coefs), speed_estimated[k, sim_n], :] @ \
                                        direction_generated_curve / np.linalg.norm(direction_curved[np.argmax(best_coefs), speed_estimated[k, sim_n], :]) / \
                                        np.linalg.norm(direction_generated_curve))

            direction_error_sphere[k, sim_n] = 1 - direction_on_sphere[np.argmax(best_coefs), speed_estimated[k, sim_n], :] @ \
                                        direction_generated_sphere / \
                                        np.linalg.norm(direction_on_sphere[np.argmax(best_coefs), speed_estimated[k, sim_n], :]) / \
                                        np.linalg.norm(direction_generated_sphere)

            direction_error_pca[k, sim_n] = 1 - abs(direction_pca[np.argmax(best_coefs), speed_estimated[k, sim_n], :] @ \
                                        direction_generated_pca / \
                                        np.linalg.norm(direction_pca[np.argmax(best_coefs), speed_estimated[k, sim_n], :]) / \
                                        np.linalg.norm(direction_generated_pca))
            print(sim_n)
        k += 1

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(snr, np.mean(direction_error_curve, axis=1), 'o-')
    plt.title('Direction detection error (with curvature)')
    plt.ylim([0, 1])
    plt.xlabel('SNR')
    plt.ylabel('1 - correlation between generated and detected')
    plt.subplot(2, 1, 2)
    plt.plot(snr, np.mean(direction_error_pca, axis=1), 'o-')
    plt.title('Direction detection error (main direction)')
    plt.xlabel('SNR')
    plt.ylim([0, 1])
    plt.ylabel('Error between detected and generated speeds in m/s')


    jitter_1 = 0.2*np.random.rand(speed_estimated.shape[1])
    jitter_2 = 0.2* np.random.rand(speed_estimated.shape[1])
    for s in range(0, speed_generated.shape[0]):
        plt.subplot(1, speed_generated.shape[0], (s+1))
        plt.scatter(speed_generated[s, :]+jitter_1, speed_estimated[s, :]+jitter_2)
        plt.xlabel('True speed, m/s')
        plt.ylabel('Estimated speed, m/s')
        plt.plot(np.arange(-1, 10, 0.1), np.arange(-1, 10, 0.1), '--')
        plt.xlim([-1, 10])
        plt.ylim([-1, 10])
        plt.title(('SNR level {0}').format(snr[s]))
        plt.xticks(range(0, 10), params['speeds'])  # Set locations and labels
        plt.yticks(range(0, 10), params['speeds'])  # Set locations and labels

    return

