```python
# Load and filter data
raw = mne.io.read_raw_fif('faces_raw.fif', preload=True)
raw.filter(1, 40)

# Set up epochs
events = mne.find_events(raw)
event_id = {'faces': 1, 'scrambled': 2}
epochs = mne.Epochs(raw, events, event_id, tmin=-0.1, tmax=0.5, baseline=(None, 0), preload=True)

# Fit ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97)
ica.fit(epochs)

# Find and remove major artifacts
ica.detect_artifacts(epochs)
ica.apply(epochs)

# Compute and visualize the contrast
evoked_faces = epochs['faces'].average()
evoked_scrambled = epochs['scrambled'].average()
contrast = mne.combine_evoked([evoked_faces, evoked_scrambled], weights=[1, -1])
contrast.plot_joint()

# Estimate noise covariance
noise_cov = mne.compute_covariance(epochs, tmax=0)

# Visualize fields on MEG helmet
evoked_faces.plot_topomap(times='peaks')

# Compute forward and inverse solutions
fwd = mne.make_forward_solution(raw.info, trans=None, src=None, bem=None)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)

# Plot the contrast in 3D
stc = mne.minimum_norm.apply_inverse(evoked_faces, inv, method='dSPM')
stc.plot(surface='white', hemi='both', subjects_dir=subjects_dir)
```