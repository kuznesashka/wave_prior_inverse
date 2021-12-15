def generate_brain_noise(G, N=500, Fs=1000):
    """Function to generate brain noise
    Parameters
    ----------
    G : numpy.ndarray
        Forward model matrix (N_channels x N_sources)
    N : int
        Number of active sources
    Fs : int
        Sampling frequency
    Returns
    -------
    brain_noise : matrix with noise (N_channels x Fs)
    """

    import numpy as np
    import mne

    Nsrc = G.shape[1]
    src_idx = np.random.randint(0, Nsrc, N)

    q = np.random.rand(N, 2 * Fs)

    alpha_band = [8, 12]
    beta_band = [15, 30]
    gamma1_band = [30, 50]
    gamma2_band = [50, 70]

    A = mne.filter.filter_data(q, Fs, alpha_band[0], alpha_band[1])
    B = mne.filter.filter_data(q, Fs, beta_band[0], beta_band[1])
    C = mne.filter.filter_data(q, Fs, gamma1_band[0], gamma1_band[1])
    D = mne.filter.filter_data(q, Fs, gamma2_band[0], gamma2_band[1])

    source_noise = (
        1 / np.mean(alpha_band) * A
        + 1 / np.mean(beta_band) * B
        + 1 / np.mean(gamma1_band) * C
        + 1 / np.mean(gamma2_band) * D
    )
    brain_noise = G[:, src_idx] @ source_noise[:, int(Fs / 2) : int(Fs + Fs / 2)] / N

    return brain_noise
