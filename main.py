from direction_error_bst import direction_error_bst, plot_roc


wave_generation_params = {
    "duration": 0.02,
    "Fs": 1000,
    "speeds": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}

data_dir = '/Users/kotkartoshka/Documents/wave_prior_data'
snr_list = [2]
spatial_jitter_list = [0]
channel_type = "grad"
simulation_number = 100

[
    roc_parameters,
    direction_error,
    direction_error_smooth,
    direction_error_pca,
    speed_simulated_array,
    speed_estimated_array,
] = direction_error_bst(
    data_dir=data_dir,
    channel_type=channel_type,
    wave_generation_params=wave_generation_params,
    snr_list=snr_list,
    spatial_jitter_list=spatial_jitter_list,
    simulation_number=simulation_number
)

plot_roc(
    spatial_jitter_list=spatial_jitter_list,
    snr_list=snr_list,
    roc_parameters=roc_parameters
)
