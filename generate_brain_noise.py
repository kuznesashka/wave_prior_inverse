import numpy as np
import mne


def generate_brain_noise(
        G: np.ndarray,
        active_source_number: int = 500,
        sampling_frequency: int = 1000,
        time_point_number: int = None
):
    """Function to generate brain noise.
    Parameters
    ----------
    G : np.ndarray
        Forward model matrix (n_channels x n_sources).
    active_source_number : int = 500
        Number of active sources.
    sampling_frequency : int = 1000
        Sampling frequency.
    time_point_number : Optional[int] = None
        Number of time samples in the resulting array. If None, is set to sampling frequency.
    Returns
    -------
    brain_noise : matrix with brain noise (n_channels x time_point_number)
    """
    total_source_number = G.shape[1]
    active_source_indices = np.random.randint(0, total_source_number, active_source_number)

    data_to_filter = np.random.rand(active_source_number, 2 * sampling_frequency)

    oscillation_band_dictionary = {
        'alpha': (8, 12),
        'beta': (15, 30),
        'gamma1': (30, 50),
        'gamma2': (50, 70)
    }

    source_noise = np.zeros(data_to_filter.shape)
    for band_name in oscillation_band_dictionary:
        band_frequency = oscillation_band_dictionary[band_name]
        data_bandpass_filtered = mne.filter.filter_data(
            data=data_to_filter,
            sfreq=sampling_frequency,
            l_freq=band_frequency[0],
            h_freq=band_frequency[1]
        )
        source_noise += 1 / np.mean(band_frequency) * data_bandpass_filtered

    brain_noise = (
            G[:, active_source_indices]
            @ source_noise[:, int(sampling_frequency / 2): int(sampling_frequency + sampling_frequency / 2)]
            / active_source_number
    )

    time_point_number = time_point_number or sampling_frequency
    brain_noise = brain_noise[:, :time_point_number]

    return brain_noise
