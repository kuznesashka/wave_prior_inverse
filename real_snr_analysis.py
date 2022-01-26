import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def calculate_snr_from_real_spikes(
        meg_file_name: str,
        spike_index_list,
        number_of_high_amplitude_channels: Optional[int] = None,
        t_min: int = 0,
        t_max: int = 600,
        sampling_frequency: int = 1000,
        channel_type: str = "grad",
        freq_low: int = 1,
        freq_high: int = 200,
        freq_notch: int = 50,
        spike_window: int = 20,
        noise_window: int = 60,
):
    meg_data = mne.io.read_raw_fif(meg_file_name)
    total_duration_ms = meg_data[0][0].shape[1]
    t_max = min(t_max, (total_duration_ms - 1) / sampling_frequency)
    meg_data.crop(t_min, t_max).load_data()

    selected_channels = mne.pick_types(meg_data.info, meg=channel_type, exclude="bads")

    meg_data.filter(freq_low, freq_high, fir_design="firwin")
    meg_data.notch_filter(freq_notch, filter_length="auto", phase="zero")
    meg_data_filtered = meg_data.get_data(picks=selected_channels)

    spike_index_list = spike_index_list[spike_index_list > 60].astype(int)
    number_of_high_amplitude_channels = number_of_high_amplitude_channels or meg_data_filtered.shape[0]

    snr_estimated = []
    for spike_i in spike_index_list:
        spike_signal = meg_data_filtered[:, (spike_i - spike_window):(spike_i + spike_window)]
        max_amplitude_channels_ind = (
            np.flip(np.argsort(np.abs(spike_signal[:, spike_window + 1])))[:number_of_high_amplitude_channels]
        )
        spike_signal_norm = np.linalg.norm(spike_signal[max_amplitude_channels_ind, :])
        noise_before = np.arange(spike_i - noise_window, spike_i - spike_window)
        noise_after = np.arange(spike_i + spike_window, spike_i + noise_window)
        noise = meg_data_filtered[max_amplitude_channels_ind, :]
        if len(np.intersect1d(noise_before, spike_i)) == 0:
            noise = noise[:, noise_before]
        elif len(np.intersect1d(noise_after, spike_i)) == 0:
            noise = noise[:, noise_after]
        else:
            continue
        noise_norm = np.linalg.norm(noise)
        snr_estimated.append(spike_signal_norm / noise_norm)
    return snr_estimated


def plot_real_spike_snr(snr_estimated: List[float], channel_type: str = "grad"):
    plt.figure()
    plt.hist(snr_estimated)
    plt.xlabel("SNR")
    plt.title(f"Real spikes SNR, {channel_type}, total spikes number = {len(snr_estimated)}")


def estimate_real_snr():
    # spikes detected automatically
    meg_file_name = "/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/B1C2_ii_run1_raw_tsss_mc_art_corr.fif"
    spike_index_list = np.loadtxt("index.csv", delimiter=",")
    snr_estimated = calculate_snr_from_real_spikes(
        meg_file_name=meg_file_name, spike_index_list=spike_index_list
    )
    plot_real_spike_snr(snr_estimated=snr_estimated)

    # spikes detected manually
    meg_file_name = "/home/ksasha/Projects/Epilepsy/OBir/sleep_raw_tsss.fif"
    spike_index_list = np.loadtxt(
        "/home/ksasha/Projects/Epilepsy/OBir/OBir_sleep_manual_2705_added.csv",
        delimiter=",",
    )
    snr_estimated = calculate_snr_from_real_spikes(
        meg_file_name=meg_file_name, spike_index_list=spike_index_list
    )
    plot_real_spike_snr(snr_estimated=snr_estimated)

