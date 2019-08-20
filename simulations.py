def simulations(data_dir, channel_type, params, snr, num_sim = 100):
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
            ROC curve plot
            """

    from create_blob_on_sensors import create_blob_on_sensors
    from create_waves_on_sensors import create_waves_on_sensors
    from generate_brain_noise import generate_brain_noise
    from LASSO_inverse_solve import LASSO_inverse_solve
    import scipy.io
    import numpy as np
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    G_raw = scipy.io.loadmat(data_dir+'/G.mat')
    cortex_raw = scipy.io.loadmat(data_dir+'/cortex.mat')

    if channel_type == 'mag':
        G = G_raw['G'][np.arange(2, 306, 3)]  # magnetometers
    elif channel_type == 'grad':
        G = G_raw['G'][np.setdiff1d(range(0, 306), np.arange(2, 306, 3))]  # gradiometers
    else:
        print('Wrong channel name')

    cortex = cortex_raw['cortex'][0]
    # vertices = cortex[0][1]

    # ntpoints = int(params['duration']*params['Fs']+1)

    y_true = np.zeros(num_sim*2)
    y_true[0:num_sim] = np.ones(num_sim)
    auc = np.zeros(len(snr))
    k = 0

    plt.figure()
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    for snr_level in snr:
        # wave_fit = np.zeros(2*Num_sim, dtype=int)
        speed_fit = np.zeros([len(snr), num_sim], dtype=int)
        direction_fit = np.zeros([len(snr), num_sim], dtype=int)
        score_fit = np.zeros(2*num_sim)
        generate_direction = np.zeros(num_sim, dtype=int)
        generate_speed = np.zeros(num_sim, dtype=int)
        src_idx = np.zeros(num_sim, dtype=int)
        brain_noise_norm = np.zeros([G.shape[0], params['Fs'],num_sim])

        # first Nsim trials with waves
        for sim_n in range(0, num_sim):
            src_idx[sim_n] = np.random.randint(0, G.shape[1])
            [sensor_waves, path_indices, path_final] = create_waves_on_sensors(cortex, params, G, src_idx[sim_n], spheric=0)

            generate_direction[sim_n] = np.random.randint(0, sensor_waves.shape[0])
            generate_speed[sim_n] = np.random.randint(0, sensor_waves.shape[1])

            # visualization
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
            # for d in range(0, path_final.shape[0]):
            #     ax.scatter(path_final[d, 10, :, 0], path_final[d, 10, :, 1], path_final[d, 10, :, 2], marker = '^')

            brain_noise = generate_brain_noise(G)
            brain_noise_norm[:, :, sim_n] = brain_noise[:, :sensor_waves.shape[3]]/np.linalg.norm(brain_noise[:, :sensor_waves.shape[3]])
            wave_picked = sensor_waves[generate_direction[sim_n], generate_speed[sim_n], :, :]
            wave_picked_norm = wave_picked/np.linalg.norm(wave_picked)
            data = snr_level*wave_picked_norm + brain_noise_norm[:, :, sim_n]

            # plt.figure()
            # plt.plot(data.T)
            [score_fit[sim_n], best_coefs, best_shift, best_speed_ind] = LASSO_inverse_solve(data, sensor_waves)
            # wave_fit[sim_n] = (score_fit[sim_n] > 0.7)
            speed_fit[k, sim_n] = (best_speed_ind == generate_speed[sim_n])
            direction_fit[k, sim_n] = (np.argmax(best_coefs) == generate_direction[sim_n])
            print(sim_n)

        # next Nsim trials without waves
        for sim_n in range(num_sim, 2*num_sim):
            idx = src_idx[sim_n-num_sim]
            [sensor_blob, path_indices] = create_blob_on_sensors(cortex, params, G, idx)
            [sensor_waves, path_indices, path_final] = create_waves_on_sensors(cortex, params, G, idx, spheric=0)

            brain_noise = brain_noise_norm[:, :, sim_n-num_sim]
            sensor_blob_norm = sensor_blob/np.linalg.norm(sensor_blob)
            data = snr_level*sensor_blob_norm + brain_noise

            # plt.figure()
            # plt.plot(data.T)

            [score_fit[sim_n], best_coefs, best_shift, best_speed_ind] = LASSO_inverse_solve(data, sensor_waves)
            # wave_fit[sim_n] = (score_fit[sim_n] > 0.7)
            print(sim_n)


        y_score = score_fit
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc[k] = metrics.roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, lw=lw, label='ROC curve for SNR {0}, (area = {1:0.2f})'.format(snr_level, auc[k]))
        k+=1

    plt.title('Receiver operating characteristics for different SNR')
    plt.legend(loc="lower right")
    plt.show()

    direction_ratio = np.zeros(len(snr))
    speed_ratio = np.zeros(len(snr))
    for i in range(0, len(snr)):
        direction_ratio[i] = sum(direction_fit[i])/num_sim*100
        speed_ratio[i] = sum(speed_fit[i])/num_sim*100

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(snr, direction_ratio, 'o-')
    plt.title('Direction detection ratio')
    plt.subplot(2,1,2)
    plt.plot(snr, speed_ratio, 'o-')
    plt.title('Speed detection ratio')

    return auc

