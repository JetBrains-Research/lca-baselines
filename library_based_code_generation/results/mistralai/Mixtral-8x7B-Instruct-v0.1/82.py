 import torch
import torchaudio
from monai.networks.nets import UNet
from monai.optimizers import SGD
from pytorch_ditto.fedprox import FedProxLoss
from pytorch_ditto.helpers import SupervisedPTDittoHelper

class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
def **init**(self, config_filename, global_epochs, personalized_epochs, task_name):
super().**init**(config_filename, global_epochs, personalized_epochs, task_name)
self.helper = SupervisedPTDittoHelper(
UNet(**self.network_params),
SGD(self.optimizer_params),
FedProxLoss(self.fedprox_params),
)

def train(self, global_model_weights, client_data_loader):
try:
global_model_weights = self.helper.get\_global\_model\_weights(global\_model\_weights)
fedprox\_loss = self.helper.prepare\_fedprox\_loss(global\_model\_weights)
personalized\_model\_info = self.helper.load\_ditto\_personalized\_model\_info()

self.helper.local\_training\_reference\_model(client\_data\_loader, fedprox\_loss)
self.helper.local\_training\_personalized\_model(client\_data\_loader, personalized\_model\_info)

return self.helper.get\_updated\_weights()
except KeyboardInterrupt:
print("Training interrupted by user")
return None
except Exception as e:
print(f"Error during training: {e}")
return None