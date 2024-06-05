  import expyfun

# Define the experiment parameters
experiment_name = 'CRM_Experiment'
window_size = 10
participant = 'John'
session = '01'
version = 'v1.0'

# Define the talker parameters
talker_1 = expyfun.Talker(gender='male', talker_number=1, sampling_rate=40000)
talker_2 = expyfun.Talker(gender='female', talker_number=1, sampling_rate=40000)

# Load the talkers
talker_1.load()
talker_2.load()

# Print the valid callsigns
print(talker_1.callsign)
print(talker_2.callsign)

# Read a sentence from the hard drive
sentence_1 = talker_1.read_sentence()
sentence_2 = talker_2.read_sentence()

# Pad and align the sentences
sentence_1_padded = expyfun.pad_sentence(sentence_1, window_size)
sentence_2_padded = expyfun.pad_sentence(sentence_2, window_size)

# Preload the talkers
talker_1.preload()
talker_2.preload()

# Get a second sentence from memory
sentence_3 = talker_1.read_sentence()
sentence_4 = talker_2.read_sentence()

# Pad and align the second sentence
sentence_3_padded = expyfun.pad_sentence(sentence_3, window_size)
sentence_4_padded = expyfun.pad_sentence(sentence_4, window_size)

# Display a text prompt on the screen
expyfun.display_text_prompt('Please respond with the number of the talker you would like to hear the sentence from.')

# Load the padded sentences into the buffer
expyfun.load_buffer(sentence_1_padded)
expyfun.load_buffer(sentence_2_padded)
expyfun.load_buffer(sentence_3_padded)
expyfun.load_buffer(sentence_4_padded)

# Start the stimulus
expyfun.start_stimulus()

# Wait for a specific duration
expyfun.wait_for_duration(10)

# Display a prompt based on the response
if expyfun.get_response() == '1':
    expyfun.display_text_prompt('You have chosen talker 1.')
elif expyfun.get_response() == '2':
    expyfun.display_text_prompt('You have chosen talker 2.')
else:
    expyfun.display_text_prompt('Invalid response. Please try again.')

# Plot a screenshot of the experiment
expyfun.plot_screenshot()