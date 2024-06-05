 import torch
from monai.networks.nets import UNet
from ptdito.helper import SupervisedPTDittoHelper
from ptdito.model import ModelName
from ptdito.model_learner import ModelLearner

class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
def **init**(self, train_config_filename, num_agg_epochs, num_ditto_model_epochs, train_task_name):
super().**init**(train_config_filename)
self.num_agg_epochs = num_agg_epochs
self.num_ditto_model_epochs = num_ditto_model_epochs
self.train_task_name = train_task_name
self.ditto_helper = SupervisedPTDittoHelper(self.model_definition, ModelName.LOCAL, self.device)

def train_config(self):
self.model = UNet(**self.model_definition).to(self.device)
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
self.ditto_helper.set_model_and_optimizer(self.model, self.optimizer)

def train(self):
def handle_abort_signal():
# handle abort signal logic
pass

while not self.should_stop:
handle_abort_signal()

# update local model weights with received weights
self.ditto_helper.update_model_weights(_receive_and_update_model())

# load Ditto personalized model
local_model = load_local_model(ModelName.PERSONALIZED)

# perform local training on the reference model and personalized model
for _ in range(self.num_ditto_model_epochs):
self.ditto_helper.train_one_epoch(local_model)

# validate the Ditto model each round
validation_results = test_a_list_of_jobs([self.ditto_helper.model], self.validation_data)

# compute the delta model
delta_model = compute_model_diff(local_model, self.ditto_helper.model)

# return a shareable object with the updated local model
return self.ditto_helper.model, delta_model, validation_results

# set up the superclass
self.train_config()

# start the training loop
self.train()