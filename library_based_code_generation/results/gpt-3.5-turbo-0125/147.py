import mne

data_path = mne.datasets.spm_face.data_path()
raw = mne.io.read_raw_nir(data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds')

raw.filter(1, 40)
events = mne.find_events(raw, stim_channel='UPPT001')

event_id = {'faces': 1, 'scrambled': 2}
epochs = mne.Epochs(raw, events, event_id, tmin=-0.1, tmax=0.5, baseline=(None, 0), preload=True)

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(epochs)
ica.plot_components()

epochs.load_data()
ica.exclude = [1, 2]
ica.apply(epochs)

epochs.average().plot()

evoked_faces = epochs['faces'].average()
evoked_scrambled = epochs['scrambled'].average()

contrast = mne.combine_evoked([evoked_faces, evoked_scrambled], weights=[1, -1])
contrast.plot_topomap()

noise_cov = mne.compute_covariance(epochs, tmax=0)

evoked_faces.plot_topomap()

fwd = mne.make_forward_solution(raw.info, trans=None, src=None, bem=None)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)

stc = mne.minimum_norm.apply_inverse(evoked_faces, inv)
stc.plot(hemi='both')