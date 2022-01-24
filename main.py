from direction_error_bst import direction_error_bst, plot_roc
from load_input_data import load_input_data
from create_waves_on_sensors import create_waves_on_sensors


wave_generation_params = {
    "duration": 0.02,
    "Fs": 1000,
    "speeds": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}

data_dir = '/Users/kotkartoshka/Documents/wave_prior_data'
snr_list = [1, 2, 3]
spatial_jitter_list = [0]
channel_type = "grad"
simulation_number = 300

(
    G_dense,
    G,
    cortex_dense,
    cortex,
    cortex_smooth_dense,
    cortex_smooth,
    vertices,
    vertices_dense,
    vertices_smooth,
    vert_conn,
    vert_normals
) = load_input_data(data_dir=data_dir, channel_type=channel_type)

start_source_index = 100000
(
    path_coordinates_array,
    path_coordinates_smooth_array,
    forward_model_updated,
    propagation_direction_coord,
    propagation_direction_coord_smooth,
    propagation_direction_coord_pca
) = create_waves_on_sensors(
    vertices=vertices,
    vertices_smooth=vertices_smooth,
    vert_conn=vert_conn,
    vert_normals=vert_normals,
    G=G,
    wave_generation_params=wave_generation_params,
    start_source_index=start_source_index,
    add_spherical_wave=False
)

# (
#     roc_parameters,
#     direction_error,
#     direction_error_smooth,
#     direction_error_pca,
#     speed_simulated_array,
#     speed_estimated_array,
# ) = direction_error_bst(
#     data_dir=data_dir,
#     channel_type=channel_type,
#     wave_generation_params=wave_generation_params,
#     snr_list=snr_list,
#     spatial_jitter_list=spatial_jitter_list,
#     simulation_number=simulation_number
# )
#
# plot_roc(
#     spatial_jitter_list=spatial_jitter_list,
#     snr_list=snr_list,
#     roc_parameters=roc_parameters
# )
