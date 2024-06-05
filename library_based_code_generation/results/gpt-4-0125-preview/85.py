```python
import numpy as np
from expyfun import ExperimentController
from expyfun.stimuli import CRMPreload, crm_sentence
from expyfun.io import write_hdf5
import os

# Experiment parameters
exp_name = 'CRM_experiment'
participant = 'subj01'
session = '001'
version = '1.0'
window_size = [800, 600]
duration = 2.0  # duration to wait for a response
sampling_rate = 40000

# Initialize CRM corpus preload
crm = CRMPreload(fs=sampling_rate)

# Get valid callsigns
print(crm.valid_callsigns)

# Prepare sentences
talker1 = crm.get_talker('male', 1)
talker2 = crm.get_talker('female', 1)

sentence1 = crm_sentence(talker1, 'charlie', 1, 2, fs=sampling_rate)
sentence2 = crm_sentence(talker2, 'charlie', 2, 3, fs=sampling_rate)

# Pad and align sentences
max_len = max(len(sentence1), len(sentence2))
sentence1_padded = np.pad(sentence1, (0, max_len - len(sentence1)), 'constant')
sentence2_padded = np.pad(sentence2, (0, max_len - len(sentence2)), 'constant')

# Start Experiment
with ExperimentController(exp_name, participant=participant, session=session, version=version,
                          window_size=window_size, full_screen=False, output_dir=os.curdir) as ec:
    ec.load_buffer(np.vstack([sentence1_padded, sentence2_padded]))

    # Display prompt
    ec.screen_prompt('Listen carefully to the sentences', duration=2)

    # Start stimulus
    ec.start_stimulus()

    # Wait for specific duration
    ec.wait_secs(duration)

    # Response menu
    response = ec.screen_menu(['Sentence 1', 'Sentence 2'], title='Which sentence did you hear first?')

    # Display prompt based on response
    if response == 0:
        ec.screen_text('You selected Sentence 1')
    else:
        ec.screen_text('You selected Sentence 2')

    ec.wait_secs(1)  # Wait a bit before ending

    # Plot a screenshot of the experiment
    screenshot = ec.screenshot()
    ec.save_image(screenshot, 'experiment_screenshot.png')
```