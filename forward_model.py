import mne
import numpy as np


meg_file_name = "/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/B1C2_ii_run1_raw_tsss_mc_art_corr.fif"
freesurfer_subject_dir = "/home/ksasha/Projects/ASPIRE_project/MEG_data/B1C2/"
subject_name = "B1C2"

"""
1. compute BEM surfaces
$ mne watershed_bem -s Bir_fs -d /home/ksasha/Projects/Epilepsy/OBir
2. check the BEM surfaces
mne.viz.plot_bem(
    subject=subject_name, 
    subjects_dir=freesurfer_subject_dir, 
    brain_surfaces='white', 
    orientation='coronal'
)
3. create scalp surfaces
$ mne make_scalp_surfaces -s Bir_fs -d /home/ksasha/Projects/Epilepsy/OBir
$ export ETS_TOOLKIT='qt4'
4. run coregistration GUI
$ mne coreg -d freesurfer_subject_dir -s subject_name
"""

trans_file = "B1C2-trans.fif"
meg_data = mne.io.read_raw_fif(fname=meg_file_name)
meg_data.load_data()

# check the coregistration
mne.viz.plot_alignment(
    info=meg_data.info,
    trans=trans_file,
    subject=subject_name,
    dig=True,
    meg=["helmet", "sensors"],
    subjects_dir=freesurfer_subject_dir,
    surfaces="head",
)

# Creating the source space
src = mne.setup_source_space(
    subject=subject_name, spacing="ico4", subjects_dir=freesurfer_subject_dir, add_dist=False
)
src_dense = mne.setup_source_space(
    subject=subject_name, spacing="ico5", subjects_dir=freesurfer_subject_dir, add_dist=False
)

# visualization of sources
# brain = Brain(subject_name, 'lh', 'inflated', subjects_dir=freesurfer_subject_dir)
# surf = brain.geo['lh']
# vertidx = np.where(src[0]['inuse'])[0]
# mlab.points3d(surf.x[vertidx], surf.y[vertidx],
#               surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)

# BEM solution
conductivity = (0.3, )  # for single layer (inner skull)
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject=subject_name, conductivity=conductivity, subjects_dir=freesurfer_subject_dir
)
bem = mne.make_bem_solution(model)

mne.viz.plot_alignment(
    info=meg_data.info,
    subject=subject_name,
    subjects_dir=freesurfer_subject_dir,
    meg="helmet",
    bem=bem,
    dig=True,
    surfaces=["brain"],
)

# forward operator
G_dense = mne.make_forward_solution(
    meg_file_name, trans=trans_file, src=src_dense, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=2
)

G = mne.make_forward_solution(
    meg_file_name, trans=trans_file, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=2
)

# leadfield = G['sol']['data']
# np.savetxt("G_OBir.csv", leadfield, delimiter=",")

mne.write_forward_solution("B1C2_fwd", fwd=G, overwrite=True)
mne.write_forward_solution("B1C2_fwd_dense", fwd=G_dense)
