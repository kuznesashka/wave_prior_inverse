from direction_error_bst import \
    direction_error_bst, \
    plot_roc, \
    plot_direction_estimation_error, \
    plot_speed_estimation_error
from load_input_data import load_input_data
from create_waves_on_sensors import create_waves_on_sensors


wave_generation_params = {
    "duration": 0.02,
    "Fs": 1000,
    "speeds": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}

data_dir = '/Users/kotkartoshka/Documents/wave_prior_data'
snr_list = [3]
spatial_jitter_list = [0]
channel_type = "grad"
simulation_num = 500

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
    vertices_smooth_dense,
    vert_conn,
    vert_conn_dense,
    vert_normals,
    vert_normals_dense
) = load_input_data(data_dir=data_dir, channel_type=channel_type)

(
    roc_parameters,
    direction_error,
    direction_error_smooth,
    direction_error_pca,
    speed_simulated_array,
    speed_estimated_array,
) = direction_error_bst(
    G=G,
    G_dense=G_dense,
    vertices=vertices,
    vertices_dense=vertices_dense,
    vertices_smooth=vertices_smooth,
    vertices_smooth_dense=vertices_smooth_dense,
    vert_conn=vert_conn,
    vert_conn_dense=vert_conn_dense,
    vert_normals=vert_normals,
    vert_normals_dense=vert_normals_dense,
    wave_generation_params=wave_generation_params,
    snr_list=snr_list,
    spatial_jitter_list=spatial_jitter_list,
    simulation_num=simulation_num,
    add_spherical_wave=False,
    plot_wave_time_series=False,
    path_length_for_blob=20,
    plot_blob_time_series=False,
    distance_to_midline=0.02
)

plot_roc(
    spatial_jitter_list=spatial_jitter_list,
    snr_list=snr_list,
    roc_parameters=roc_parameters
)

plot_direction_estimation_error(
    snr_list=snr_list,
    spatial_jitter_list=spatial_jitter_list,
    direction_error=direction_error,
    direction_error_smooth=direction_error_smooth,
    direction_error_pca=direction_error_pca,
)

plot_speed_estimation_error(
    snr_list=snr_list,
    spatial_jitter_list=spatial_jitter_list,
    speed_simulated_array=speed_simulated_array,
    speed_estimated_array=speed_estimated_array,
    simulation_num=simulation_num,
    wave_generation_params=wave_generation_params
)

# start_source_index = 100000
# (
#     waves_on_sensors,
#     direction_final,
#     direction_final_smooth,
#     direction_pca
# ) = create_waves_on_sensors(
#     vertices=vertices,
#     vertices_smooth=vertices_smooth,
#     vert_conn=vert_conn,
#     vert_normals=vert_normals,
#     G=G,
#     wave_generation_params=wave_generation_params,
#     start_source_index=start_source_index,
#     add_spherical_wave=False,
#     plot_time_series=False
# )
