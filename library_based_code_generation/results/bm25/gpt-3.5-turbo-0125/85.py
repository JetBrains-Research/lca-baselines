import expyfun as ef

talker1 = ef.crm_prepare_corpus('CRM', talker='female', number=1, sampling_rate=40000)
talker2 = ef.crm_prepare_corpus('CRM', talker='male', number=1, sampling_rate=40000)

print(talker1.valid_callsigns)
print(talker2.valid_callsigns)

sentence1 = ef.crm_sentence('path_to_sentence1.wav')
sentence2 = ef.crm_sentence('path_to_sentence2.wav')

padded_sentence1 = ef.sentence(sentence1, align='start')
padded_sentence2 = ef.sentence(sentence2, align='start')

ef.CRMPreload([padded_sentence1, padded_sentence2])

exp = ef.ExperimentController('experiment_name', window_size=0.2, participant='John Doe', session=1, version=1)

exp.screen_prompt('Press any key to start the experiment')

ef.load_buffer([padded_sentence1, padded_sentence2])
ef.start_stimulus()
ef.wait(5)

response = ef.crm_response_menu(['Yes', 'No'])
if response == 'Yes':
    exp.screen_text('Correct response!')
else:
    exp.screen_text('Incorrect response!')

ef.plot_screen()