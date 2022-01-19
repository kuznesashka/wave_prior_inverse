from simulations_roc import simulations
from direction_error_bst import direction_error_bst

from typing import List, Dict, Any


wave_generation_params = {
    "duration": 0.02,
    "Fs": 1000,
    "speeds": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}


def create_simulations_for_paper(
        data_dir: str,
        wave_generation_params: Dict[str, Any],
        snr_list: List[float],
        spatial_jitter_list: List[float],
        channel_type: str = "grad",
        simulation_number: int = 100
):
    """
    simulations from Fig. 3 (https://www.biorxiv.org/content/10.1101/2020.05.17.101121v1.full.pdf)

    Returns
    -------

    """

    [
        auc,
        direction_error,
        direction_error_smooth,
        direction_error_pca,
        speed_generated,
        speed_estimated,
    ] = direction_error_bst(
        data_dir=data_dir,
        channel_type=channel_type,
        wave_generation_params=wave_generation_params,
        snr_list=snr_list,
        spatial_jitter_list=spatial_jitter_list,
        simulation_number=simulation_number
    )

    return (
        auc,
        direction_error,
        direction_error_smooth,
        direction_error_pca,
        speed_generated,
        speed_estimated
    )
