import numpy as np
from typing import List, Dict, Any

from create_waves_on_sensors import create_waves_on_sensors
from generate_brain_noise import generate_brain_noise
from LASSO_inverse_solve import lasso_inverse_solve
from direction_error_bst import pick_random_source, add_brain_noise_to_signal


def calculate_speed_error_wo_spatial_bias(
        wave_generation_parameters: Dict[str, Any],
        G_dense: np.ndarray,
        vertices_dense: np.ndarray,
        vertices_dense_smooth,
        vert_conn_dense,
        vert_normals_dense,
        snr_list: List[float],
        simulation_num: int = 100,
        distance_to_midline: float = 0.02
):
    """Run Monte Carlo simulations without any spatial error.
    Control only speed detection error.

    Parameters
    ----------
    wave_generation_parameters
    G_dense
    vertices_dense
    vertices_dense_smooth
    vert_conn_dense
    vert_normals_dense
    snr_list
    simulation_num
    distance_to_midline

    Returns
    -------
    speed_simulated_array
    speed_estimated_array
    """
    snr_num = len(snr_list)
    speed_list = wave_generation_parameters["speeds"]
    speed_num = len(speed_list)
    total_duration = int(wave_generation_parameters["duration"] * wave_generation_parameters["Fs"] + 1) * 2

    speed_simulated_array = np.zeros([snr_num, simulation_num], dtype=int)
    speed_estimated_array = np.zeros([snr_num, simulation_num], dtype=int)

    for snr_i, snr in enumerate(snr_list):
        print(f"calculations for SNR = {snr} of {snr_list}")
        r_squared_per_simulation = []
        starting_source_dense_list = []

        for simulation_i in range(simulation_num):
            source = pick_random_source(vertices=vertices_dense, distance_to_midline=distance_to_midline)
            starting_source_dense_list.append(source)

            waves_on_sensors, *_ = create_waves_on_sensors(
                vertices=vertices_dense,
                vertices_smooth=vertices_dense_smooth,
                vert_conn=vert_conn_dense,
                vert_normals=vert_normals_dense,
                G=G_dense,
                wave_generation_params=wave_generation_parameters,
                start_source_index=source,
                add_spherical_wave=False,
                plot_time_series=False,
            )

            speed_simulated_index = np.random.randint(speed_num)
            speed_simulated_array[snr_i, simulation_i] = speed_simulated_index

            direction_num = waves_on_sensors.shape[0]
            direction_ind = np.random.randint(direction_num)

            brain_noise = generate_brain_noise(G=G_dense, time_point_number=total_duration)
            wave_selected = waves_on_sensors[direction_ind, speed_simulated_index, :, :]

            data = add_brain_noise_to_signal(
                signal_on_sensors=wave_selected,
                brain_noise=brain_noise,
                snr=snr,
            )

            waves_on_sensors, *_ = create_waves_on_sensors(
                vertices=vertices_dense,
                vertices_smooth=vertices_dense_smooth,
                vert_conn=vert_conn_dense,
                vert_normals=vert_normals_dense,
                G=G_dense,
                wave_generation_params=wave_generation_parameters,
                start_source_index=source,
                add_spherical_wave=False,
                plot_time_series=False,
            )

            score, speed_ind, *_ = lasso_inverse_solve(
                signal_data=data,
                wave_data=waves_on_sensors,
                fit_intercept=False
            )

            r_squared_per_simulation.append(score)
            speed_estimated_array[snr_i, simulation_i] = speed_ind

            print(f"{simulation_i} out of {simulation_num} completed")

    return speed_simulated_array, speed_estimated_array
