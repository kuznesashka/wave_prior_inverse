import mne
import numpy as np
import matplotlib.pyplot as plt

fname = '/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/B1C2_ii_run1_raw_tsss_mc_art_corr.fif'  # path to MEG
subjects_dir = '/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/'  # The paths to Freesurfer reconstructions
subject = 'B1C2'

Data = mne.io.read_raw_fif(fname)  # upload the data
tmin = 0
tmax = min(600, (Data[0][0].shape[1]-1)/1000)
Data.crop(tmin, tmax).load_data()

picks = mne.pick_types(Data.info, meg='grad', exclude='bads')

Data.filter(1, 200, fir_design='firwin')
Data.notch_filter(50, filter_length='auto', phase='zero')

spike_ind = np.loadtxt('index.csv', delimiter=',')
F = Data.get_data(picks=picks)

spike_ind = spike_ind[spike_ind > 60].astype(int)
snr = np.zeros(len(spike_ind))
for i in range(0, len(spike_ind)):
    spike = F[:, (spike_ind[i]-20):(spike_ind[i]+20)]
    signal_norm = np.linalg.norm(spike)
    noise_before = np.arange((spike_ind[i]-60), (spike_ind[i]-20))
    noise_after = np.arange((spike_ind[i] + 20), (spike_ind[i] + 60))
    if len(np.intersect1d(noise_before, spike_ind)) == 0:
        noise = F[:, noise_before]
    elif len(np.intersect1d(noise_after, spike_ind)) == 0:
        noise = F[:, noise_after]
    else:
        continue
    noise_norm = np.linalg.norm(noise)
    snr[i] = signal_norm/noise_norm

plt.figure()
plt.hist(snr)
plt.xlabel('SNR')
plt.title(['Real spikes SNR, grads, total number', len(spike_ind)])
