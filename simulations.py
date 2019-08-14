import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

G_raw = scipy.io.loadmat('/home/ksasha/PycharmProjects/Wave_prior/Wave_prior_inverse/G.mat')
cortex_raw = scipy.io.loadmat('/home/ksasha/PycharmProjects/Wave_prior/Wave_prior_inverse/cortex.mat')
G = G_raw['G'][np.arange(2, 306, 3)]  # magnetometers
cortex = cortex_raw['cortex'][0]
vertices = cortex[0][1]

# wave generation parameters
params = {'duration': 0.02, 'Fs': 1000, 'speeds': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}

Num_sim = 100
snr_level = 5

wave_fit = np.zeros(Num_sim, dtype=int)
speed_fit = np.zeros(Num_sim, dtype=int)
direction_fit = np.zeros(Num_sim, dtype=int)
generate_direction = np.zeros(Num_sim, dtype=int)
generate_speed = np.zeros(Num_sim, dtype=int)

for sim_n in range(0, Num_sim):
    src_idx = np.random.randint(0, G.shape[1])
    # sensor_waves = [directions_number x number_of_speeds x channels_number x timepoints_number]
    [sensor_waves, path_indices, path_final] = create_waves_on_sensors(cortex, params, G, src_idx, spheric=0)

    generate_direction[sim_n] = np.random.randint(0, sensor_waves.shape[0])
    generate_speed[sim_n] = np.random.randint(0, sensor_waves.shape[1])

    # visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
    # for d in range(0, path_final.shape[0]):
    #     ax.scatter(path_final[d, 10, :, 0], path_final[d, 10, :, 1], path_final[d, 10, :, 2], marker = '^')

    brain_noise = generate_brain_noise(G)
    brain_noise_norm = brain_noise/np.linalg.norm(brain_noise)
    wave_picked = sensor_waves[generate_direction[sim_n], generate_speed[sim_n], :, :]
    wave_picked_norm = wave_picked/np.linalg.norm(wave_picked)

    data = snr_level*wave_picked_norm + brain_noise_norm[:,:sensor_waves.shape[3]]
    [best_score, best_coefs, best_shift, best_speed_ind] = LASSO_inverse_solve(data, sensor_waves)

    wave_fit[sim_n] = (best_score > 0.7)
    speed_fit[sim_n] = (best_speed_ind == generate_speed[sim_n])
    direction_fit[sim_n] = (np.argmax(best_coefs) == generate_direction[sim_n])
    print(sim_n)

