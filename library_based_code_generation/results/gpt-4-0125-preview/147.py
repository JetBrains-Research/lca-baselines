```python
import mne
from mne.datasets import spm_face
from mne.preprocessing import ICA
from mne.minimum_norm import make_inverse_operator, apply_inverse

# Load data
data_path = spm_face.data_path()
raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds'
raw = mne.io.read_raw_ctf(raw_fname, preload=True)

# Filter data
raw.filter(1, 30, fir_design='firwin')

# Set up and fit ICA
ica = ICA(n_components=0.95, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [0]  # assume the first component is eye movement and exclude it
ica.apply(raw)

# Epoching
events = mne.find_events(raw, stim_channel='UPPT001')
event_id = {'faces': 1, 'scrambled': 2}
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, baseline=(None, 0),
                    preload=True)
epochs.pick_types(meg=True)  # only use MEG channels

# Average epochs
evoked_faces = epochs['faces'].average()
evoked_scrambled = epochs['scrambled'].average()
contrast = mne.combine_evoked([evoked_faces, evoked_scrambled], weights=[1, -1])

# Compute noise covariance
noise_cov = mne.compute_covariance(epochs, tmax=0., method='shrunk')

# Visualize the contrast
contrast.plot_joint()

# Compute forward model
info = evoked_faces.info
trans = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D_raw-trans.fif'
src = mne.setup_source_space(subject='fsaverage', spacing='oct6',
                             subjects_dir=data_path + '/subjects', add_dist=False)
model = mne.make_bem_model(subject='fsaverage', ico=4,
                           subjects_dir=data_path + '/subjects')
bem = mne.make_bem_solution(model)
fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1)

# Visualize sensor alignment
mne.viz.plot_alignment(info, trans, subject='fsaverage', src=src, bem=bem,
                       subjects_dir=data_path + '/subjects', dig=True,
                       meg=['helmet', 'sensors'], eeg=False)

# Compute inverse solution
inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse(contrast, inverse_operator, lambda2, method=method, pick_ori=None)

# Plot the contrast in 3D
brain = stc.plot(subjects_dir=data_path + '/subjects', subject='fsaverage',
                 hemi='both', views='lateral', initial_time=0.1, time_unit='s',
                 size=(800, 400), smoothing_steps=10)
```