 import torch
import torch.nn as nn
from monai.networks.nets import UNet
from ditto.supervised.helper import SupervisedPTDittoHelper
from ditto.supervised.training import SupervisedDittoTraining
from ditto.supervised.training_command_module import SupervisedDittoTrainingCommandModule
from ditto.supervised.training_topic import TrainingTopic
from ditto.supervised.global_model_eval import GlobalModelEval
from ditto.supervised.model_name import ModelName
from ditto.supervised.comm_a import CommA
from typing import Dict, Any

class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
def **init**(self, training_config_filename: str, num\_global\_epochs: int,
num\_personalized\_epochs: int, task\_name: str):
super().**init**(training\_config\_filename, num\_global\_epochs,
num\_personalized\_epochs, task\_name)
self.helper = SupervisedPTDittoHelper(
model=UNet(spatial\_dims=3, in\_channels=1, out\_channels=1),
optimizer=nn.SGD(self.helper.model.parameters(), lr=0.01),
)

def training\_configuration(self) -> TrainingTopic:
return TrainingTopic(
training\_command\_module=SupervisedDittoTrainingCommandModule(
model\_name=ModelName.PROSTATE,
num\_global\_epochs=self.num\_global\_epochs,
num\_personalized\_epochs=self.num\_personalized\_epochs,
),
)

def training(self, weights: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
global\_model\_weights = self._get\_model\_weights(weights, "global")

self.helper.prepare\_for\_fedprox\_loss(global\_model\_weights)

personalized\_model\_info = self._receive\_and\_update\_model(
should\_connect\_to\_server=should\_connect\_to\_server(),
should\_be\_checked=should\_be\_checked(),
get\_number\_of\_bins=get\_number\_of\_bins(),
get\_model\_for\_validation=get\_model\_for\_validation(),
model\_weights=weights,
model\_name=ModelName.PROSTATE,
)

try:
self.helper.load\_ditto\_personalized\_model\_info(personalized\_model\_info)

updated\_weights = self.helper.train(
num\_epochs=self.num\_personalized\_epochs,
)
except Exception as e:
self.logger.exception(f"Training failed with exception: {e}")
raise
except KeyboardInterrupt:
self.logger.info("Training was interrupted by user")
self.stop\_training = True
return {}

return {"global": updated\_weights}