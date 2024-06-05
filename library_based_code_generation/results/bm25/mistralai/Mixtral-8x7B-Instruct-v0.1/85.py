import expyfun as exp

corpus = exp.crm_prepare\_corpus("CRM", samplerate=40000, talkers=["b001m", "b001f"])
callsigns = corpus.get\_callsigns()
print("Valid callsigns:", callsigns)

sentence1 = corpus.get\_sentence("b001m", 1)
sentence2 = corpus.get\_sentence("b001f", 1)

padded\_sentence1 = exp.sentence(sentence1, corpus.get\_durations()[0])
padded\_sentence2 = exp.sentence(sentence2, corpus.get\_durations()[1])

exp\_name = "My Experiment"
window\_size = (800, 600)
participant = "subj001"
session = "session001"
version = "1.0"

controller = exp.ExperimentController(exp\_name, window\_size, participant, session, version)

response\_menu = exp.crm\_response\_menu(callsigns)

screen\_prompt(controller, "Get ready to start the experiment!")

exp.load\_buffer(controller, padded\_sentence1)
exp.load\_buffer(controller, padded\_sentence2)

exp.start\_stimulus(controller)
exp.wait\_for\_click\_on(controller)

duration = 5000 # 5 seconds
exp.run\_subprocess("sleep", [duration / 1000])

exp.clear\_display(controller)
exp.load\_buffer(controller, padded\_sentence2)
exp.load\_buffer(controller, padded\_sentence1)

exp.start\_stimulus(controller)
exp.wait\_for\_click\_on(controller)

sentence3 = corpus.get\_sentence("b001f", 2)
exp.load\_buffer(controller, exp.sentence(sentence3, corpus.get\_durations()[2]))

exp.screen\_text(controller, "Please respond:", response\_menu)
exp.wait\_for\_click\_on(controller)

response = exp.get\_response(controller)
exp.screen\_prompt(controller, f"You responded: {response}")

exp.plot\_screen(controller)
exp.save\_screen(controller)

exp.quit\_on\_close(controller)