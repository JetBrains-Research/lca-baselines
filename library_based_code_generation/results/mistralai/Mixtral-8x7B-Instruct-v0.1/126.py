 class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
def **init**(self, config_file, num_agg_epochs, num_ditto_epochs, task_name):
super().**init**(config_file)
self.num_agg_epochs = num_agg_epochs
self.num_ditto_epochs = num_ditto_epochs
self.task_name = task_name
self.ditto_helper = SupervisedPTDittoHelper()

def train_config(self):
model = UNet(**self.model_config)
optimizer = Adam(model.parameters(), lr=self.learning_rate)
self.ditto_helper.set_model_and_optimizer(model, optimizer)

def train(self):
while True:
abort = False
try:
received_weights = self.ditto_helper.get_received_weights()
if received_weights is not None:
self.ditto_helper.update_local_weights(received_weights)
personalized_model = self.ditto_helper.load_personalized_model()
round_outputs = []
for epoch in range(self.num_ditto_epochs):
self.ditto_helper.set_current_round(epoch)
local_train_outputs = self.local_train()
personalized_train_outputs = personalized_model.train()
round_outputs.extend([local_train_outputs, personalized_train_outputs])
validation_output = self.validate()
round_outputs.append(validation_output)
delta_model = self.compute_delta_model()
shareable_object = self.create_shareable_object(round_outputs, delta_model)
except AbortSignal:
abort = True
if not abort:
self.ditto_helper.send_shareable_object(shareable_object)