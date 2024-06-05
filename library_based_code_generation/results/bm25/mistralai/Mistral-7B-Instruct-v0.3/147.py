 Here is a simplified Python code snippet that demonstrates the pipeline you described using the MNE-Python library. Please note that this is a simplified version and might need adjustments based on your specific setup and requirements.

```python
import mne
import numpy as np
import matplotlib.pyplot as plt

# Load data
raw = mne.io.read_raw_fif('SPM_Faces.fif', preload=True)

# Filter data
raw.filter(l_freq=1, h_freq=40)

# Set up epochs
event_id = {'faces': 1, 'scrambled': 2}
epochs = mne.Epochs(raw, events=raw.info['events'], event_id=event_id, tmin=-0.2, tmax=0.6, baseline=(None, 0))

# Fit ICA and remove artifacts
ica = mne.preprocessing.ICA(n_components=None)
ica.fit(epochs.data)
ica_names = ica.get_names(epochs.info)
for name in ['ADJUST', 'EMOTION', 'EYEBLINK', 'EYEMOVEMENT', 'MUSCULAR', 'OCULOMOTOR', 'RESPIRATION']:
    if name in ica_names:
        ica.exclude = np.where(np.array(ica_names) == name)[0]

# Compute and visualize the contrast
contrast_indices = _get_contrast_indices(epochs, 'faces', 'scrambled')
averaged_epochs = epochs[contrast_indices].average()

# Estimate noise covariance
noise_cov = compute_covariance(averaged_epochs, method='welch')

# Compute forward and inverse solutions
forward_sol = mne.minimum_norm.make_forward_solution(raw.info)
inverse_sol = mne.minimum_norm.make_inverse_operator(forward_sol, noise_cov=noise_cov)

# Compute and visualize fields on MEG helmet
mne.viz.plot_topomap(averaged_epochs.data, titles=['Raw data'], show=False, scalp=True)

# Compute source reconstruction using dSPM
dspm = mne.minimum_norm.DSPM(raw.info, forward_sol, inverse_sol)
dspm_results = dspm.fit(averaged_epochs)

# Compute and plot the contrast in 3D
fig = plt.figure(figsize=(8, 6))
mne.viz.plot_3d_spectrum(dspm_results, titles=['dSPM'], show=False)
plt.show()
```

This code assumes that you have the SPM Faces dataset in a .fif format and that you have the MNE-Python library installed. The `_get_contrast_indices` function is not provided in this code snippet, as it would depend on the specific event structure of your dataset. You may need to adjust the artifact names and the event IDs according to your dataset.