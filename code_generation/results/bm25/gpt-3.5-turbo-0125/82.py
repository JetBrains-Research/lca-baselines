```python
from monai.supervised import SupervisedMonaiProstateLearner
from monai.supervised import SupervisedPTDittoHelper
from monai.networks.nets import UNet
from monai.optimizers import SGD

class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(self, training_config_filename, global_epochs, personalized_epochs, task_name):
        super().__init__()
        self.training_config_filename = training_config_filename
        self.global_epochs = global_epochs
        self.personalized_epochs = personalized_epochs
        self.task_name = task_name

    def training_configuration(self):
        self.helper = SupervisedPTDittoHelper(model=UNet(), optimizer=SGD())

    def training(self):
        # Pipeline for Ditto training
        global_model_weights = self.helper.get_global_model_weights()
        fedprox_loss = self.helper.prepare_fedprox_loss()
        ditto_personalized_model_info = self.helper.load_ditto_personalized_model_info()
        reference_model_weights = self.helper.local_training_reference_model(global_model_weights, fedprox_loss)
        personalized_model_weights = self.helper.local_training_personalized_model(ditto_personalized_model_info, reference_model_weights)
        updated_reference_model_weights = self.helper.return_updated_reference_model_weights(reference_model_weights)

        # Handle abort signals and exceptions
        self.helper.handle_abort_signals_and_exceptions()
```