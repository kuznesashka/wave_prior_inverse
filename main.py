from simulations import simulations

# Simulations

channel_type = 'grad'
data_dir = '/home/ksasha/PycharmProjects/Wave_prior_inverse'
params = {'duration': 0.02, 'Fs': 1000, 'speeds': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]} # wave generation parameters
num_sim = 100
snr = [0.1, 0.2, 0.5, 1, 2, 3]

auc = simulations(data_dir, channel_type, params, snr, num_sim)