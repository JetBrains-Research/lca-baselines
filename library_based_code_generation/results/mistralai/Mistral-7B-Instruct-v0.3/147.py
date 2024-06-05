 Here is a simplified version of the code that follows your instructions. Please note that this is a basic example and might need adjustments to fit your specific needs. Also, you need to have MNE-Python installed.

```python
import mne
import mne.preprocessing
import mne.minimum_norm
import mne.viz
import numpy as np

# Load data
raw = mne.io.read_raw_fif('SPM_Faces_MEG.fif', preload=True)

# Filter data
raw.filter(l_freq=1, h_freq=40)

# Set up epochs
event_id = {'Face': 1, 'Scrambled': 2}
events = mne.find_events(raw, stim_channel='STC 012', event_id=event_id)
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.6, baseline=(None, -0.2))

# Fit ICA and remove artifacts
ica = mne.preprocessing.ICA(n_components=None)
ica.fit(epochs)
ica_names = ica.get_names(epochs)
artifact_channels = ['MEG 012', 'MEG 013', 'MEG 014']
for channel in artifact_channels:
    if channel in ica_names:
        ica_names.remove(channel)
epochs_clean = ica.apply(epochs)

# Find and remove major artifacts
epochs_clean.pick_types(meg=True)
epochs_clean.exclude(ch_name='MEG 012')

# Compute and visualize the contrast
contrast = epochs_clean['Face'] - epochs_clean['Scrambled']
contrast.plot()

# Estimate noise covariance
noise_cov = mne.preprocessing.compute_covariance(epochs_clean)

# Visualize fields on MEG helmet
mne.viz.plot_raw_topomap(raw, tmin=-0.2, tmax=0.6, titles=['Raw MEG'], show=False, return_fig=True)
mne.viz.plot_topomap(contrast, titles=['Contrast'], show=False)

# Compute forward and inverse solutions
forward_sol = mne.minimum_norm.read_forward_solution('SPM_Faces_MEG.fif')
inverse_sol = mne.minimum_norm.make_inverse_operator(forward_sol, noise_cov=noise_cov, regularization='none')

# Source reconstruction using dSPM on the contrast
src = mne.read_source_spaces('SPM_Faces_fsaverage')
stc = mne.minimum_norm.apply_inverse(contrast, inverse_sol, src, verbose='error', return_stc=True)

# Plot the contrast in 3D
mne.viz.plot_3d(stc, subjects_dir='/path/to/subjects', title='dSPM', show=False)
```

This code assumes that you have the SPM Faces dataset in the format of a .fif file, and the source space is available in the format of an .fif file as well. You may need to adjust the paths and parameters according to your specific setup.