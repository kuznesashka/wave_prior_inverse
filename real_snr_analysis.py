import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
import ASPIRE_all_funcs as aspire
from mayavi import mlab
from surfer import Brain

fname = '/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/B1C2_ii_run1_raw_tsss_mc_art_corr.fif'  # path to MEG
subjects_dir = '/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/'  # The paths to Freesurfer reconstructions
subject = 'B1C2'

Data = mne.io.read_raw_fif(fname)  # upload the data
tmin = 0
tmax = min(600, (Data[0][0].shape[1]-1)/1000)
Data.crop(tmin, tmax).load_data()

picks = mne.pick_types(Data.info, meg='mag', exclude='bads')

Data.filter(1, 200, fir_design='firwin')
Data.notch_filter(50, filter_length='auto', phase='zero')

# Data.plot(lowpass=70)

method = 'fastica'  # calculate ICA components to delete artifacts
n_components = 40
decim = 3
random_state = 23
ica = ICA(n_components=n_components, method=method, random_state=random_state)
reject = dict(mag=5e-12, grad=4000e-13)
ica.fit(Data, start=0, stop=600., decim=decim, reject=reject)

ica.plot_components(picks=range(0,40), ch_type='mag')
ica.plot_sources(Data, picks=range(0,40))
[ecg_comp, ecg_corr] = ica.find_bads_ecg(Data)
ica.apply(Data, exclude=ecg_comp)

fwd = mne.read_forward_solution('case_1001_fwd')
# leadfield = np.loadtxt('G_OBir.csv', delimiter=",")
leadfield = fwd['sol']['data']
G2 = func.G3toG2(leadfield, picks)

locs = np.empty((0,3))
for src_hemi in fwd['src']:
    vertidx = src_hemi['vertno']
    locs = np.append(locs, src_hemi['rr'][vertidx,:], axis=0)

# spike_marks = np.loadtxt('OBir_sleep_manual_2705_added.csv', delimiter=',')

fmin = 2
fmax = 70
n_components = 40
decision = 0.95
spike_ind = func.SpikeDetect(Data, G2, fmin, fmax, picks, n_components, method,
                             decim, random_state, decision)
spike_ind = spike_ind.astype(int)

[ValMax, IndMax] = func.spike_localization(spike_ind, Data, G2, picks)
plt.hist(ValMax)

corr_thresh = 0.995
ind_m = np.nonzero(ValMax>corr_thresh)[0]
print('Number of spike found:', len(ind_m))

Nmin = 8
thr_dist = 0.01
cluster = func.clustering(spike_ind, G2, Nmin, ValMax, IndMax, ind_m, locs, thr_dist)

# visualization of sources

brain = Brain(subject, 'both', 'inflated', alpha=1, subjects_dir=subjects_dir)
surflh = brain.geo['lh']
surfrh = brain.geo['rh']
num_left = fwd['src'][0]['nuse']
vertidxlh = np.where(fwd['src'][0]['inuse'])[0]
vertidxrh = np.where(fwd['src'][1]['inuse'])[0]

# colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980),
#           (0.9290, 0.6940, 0.1250), (0.4940, 0.1840, 0.5560),
#           (0.4660, 0.6740, 0.1880), (0.3010, 0.7450, 0.9330),
#           (0.6350, 0.0780, 0.1840), (0.15, 0.15, 0.15),
#           (0, 0.5, 0.5), (0.3, 0.18, 0.4),
#           (0.8, 0.08, 0.5), (0.9, 0.5, 0.5), (0.1, 0.3, 0.2), (0.1, 0.3, 0.2)]

for i in range(len(cluster)):
    ind = cluster[i][:, 0].astype(int)
    mlab.points3d(surflh.x[vertidxlh[ind[ind<num_left]]], surflh.y[vertidxlh[ind[ind<num_left]]],
                  surflh.z[vertidxlh[ind[ind<num_left]]], color=(1,1,0), scale_factor=1.5)
    mlab.points3d(surfrh.x[vertidxrh[ind[ind>=num_left]-num_left]],
                  surfrh.y[vertidxrh[ind[ind>=num_left]-num_left]],
                  surfrh.z[vertidxrh[ind[ind>=num_left]-num_left]], color=(1,1,0),
                  scale_factor=1.5)
