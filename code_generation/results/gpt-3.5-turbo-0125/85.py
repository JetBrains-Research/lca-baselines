import expyfun

talker1 = expyfun.get_talker('CRM', gender='male', talker_number=1, sampling_rate=40000)
talker2 = expyfun.get_talker('CRM', gender='female', talker_number=1, sampling_rate=40000)

print(talker1.valid_callsigns)

sentence1 = expyfun.read_sentence_from_disk('path_to_sentence1.wav')
sentence2 = expyfun.read_sentence_from_memory('sentence2.wav')

padded_sentence1, padded_sentence2 = expyfun.pad_and_align_sentences(sentence1, sentence2)

expyfun.preload_talkers([talker1, talker2])

expyfun.run_experiment(name='my_experiment', window_size=100, participant='John Doe', session=1, version=1)

expyfun.display_text_prompt('Press any key to start the experiment')

expyfun.load_sentences_into_buffer([padded_sentence1, padded_sentence2])

expyfun.start_stimulus()

expyfun.wait_for_duration(10)

response = expyfun.response_menu()

if response == 'correct':
    expyfun.display_prompt('Correct response!')
else:
    expyfun.display_prompt('Incorrect response!')

expyfun.plot_screenshot()