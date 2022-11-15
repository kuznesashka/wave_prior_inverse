from direction_error_bst import direction_error_bst

# Simulations
channel_type = "grad"
data_dir = "wave_prior_inverse"

# wave generation parameters
params = {
    "duration": 0.02,
    "Fs": 1000,
    "speeds": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}
num_sim = 100
snr = [2]
spatial_jitter = [0]

[
    auc,
    direction_error,
    direction_error_smooth,
    direction_error_pca,
    speed_generated,
    speed_estimated,
] = direction_error_bst(data_dir, channel_type, params, snr, spatial_jitter, num_sim)
