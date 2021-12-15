import mne
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fname = "/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/B1C2_ii_run1_raw_tsss_mc_art_corr.fif"  # path to MEG
subjects_dir = "/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/"  # The paths to Freesurfer reconstructions
subject = "B1C2"

# BEM surfaces and coregistration
# compute BEM surfaces
# $ mne watershed_bem -s Bir_fs -d /home/ksasha/Projects/Epilepsy/OBir
# check the BEM surfaces
# mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
#                   brain_surfaces='white', orientation='coronal')

# in terminal
# create surfaces
# $ mne make_scalp_surfaces -s Bir_fs -d /home/ksasha/Projects/Epilepsy/OBir
# $ export ETS_TOOLKIT='qt4'
# coregistration GUI
# $ mne coreg -d subjects_dir -s subject
# compute BEM surfaces
# trans.fif -- the transformation file after coregistration

trans = "B1C2-trans.fif"
Data = mne.io.read_raw_fif(fname=fname)
Data.load_data()

# check the coregistration
mne.viz.plot_alignment(
    Data.info,
    trans,
    subject=subject,
    dig=True,
    meg=["helmet", "sensors"],
    subjects_dir=subjects_dir,
    surfaces="head",
)

# Creating the source space
src = mne.setup_source_space(
    subject, spacing="ico4", subjects_dir=subjects_dir, add_dist=False
)
src_dense = mne.setup_source_space(
    subject, spacing="ico5", subjects_dir=subjects_dir, add_dist=False
)

# visualization of sources
# brain = Brain(subject, 'lh', 'inflated', subjects_dir=subjects_dir)
# surf = brain.geo['lh']
# vertidx = np.where(src[0]['inuse'])[0]
# mlab.points3d(surf.x[vertidx], surf.y[vertidx],
#               surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)

# BEM solution
conductivity = (0.3,)  # for single layer (inner skull)
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject=subject, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)

mne.viz.plot_alignment(
    Data.info,
    subject=subject,
    subjects_dir=subjects_dir,
    meg="helmet",
    bem=bem,
    dig=True,
    surfaces=["brain"],
)

# forward operator
fwd_dense = mne.make_forward_solution(
    fname,
    trans=trans,
    src=src_dense,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=2,
)

fwd = mne.make_forward_solution(
    fname, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=2
)

# leadfield = fwd['sol']['data']
# np.savetxt("G_OBir.csv", leadfield, delimiter=",")
mne.write_forward_solution("B1C2_fwd", fwd=fwd, overwrite=True)
mne.write_forward_solution("B1C2_fwd_dense", fwd=fwd_dense)
