  ```
from monai.transforms import *
from monai.networks import *
from monai.optimizers import *
from monai.losses import *
from monai.metrics import *
from monai.utils import *
from monai.config import *
from monai.engine import *
from monai.engine.train_ditto import *
from monai.engine.train_ditto.utils import *
from monai.engine.train_ditto.training_loop import *
from monai.engine.train_ditto.training_topic import *
from monai.engine.train_ditto.stop_training import *
from monai.engine.train_ditto.training_step import *
from monai.engine.train_ditto.test_a_list_of_jobs import *
from monai.engine.train_ditto._process_training_result import *
from monai.engine.train_ditto._prepare_training_ctx import *
from monai.engine.train_ditto.training_command_module import *
from monai.engine.train_ditto.comm_a import *
from monai.engine.train_ditto.get_model_for_validation import *
from monai.engine.train_ditto._receive_and_update_model import *
from monai.engine.train_ditto.should_be_checked import *
from monai.engine.train_ditto.get_number_of_bins import *
from monai.engine.train_ditto._check_dict_b_value_same_as_dict_a_for_keys_in_dict_a import *
from monai.engine.train_ditto._get_model_weights import *
from monai.engine.train_ditto.save_model_global import *
from monai.engine.train_ditto.global_model_eval import *
from monai.engine.train_ditto.model_name import *
from monai.engine.train_ditto.should_connect_to_server import *
from monai.engine.train_ditto._wait_for_task import *

class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(self, training_config_filename, num_global_epochs, num_personalized_epochs, task_name):
        super().__init__(training_config_filename, num_global_epochs, num_personalized_epochs, task_name)
        self.training_config_filename = training_config_filename
        self.num_global_epochs = num_global_epochs
        self.num_personalized_epochs = num_personalized_epochs
        self.task_name = task_name

    def training_config(self):
        self.training_config_helper = SupervisedPTDittoHelper(self.model, self.optimizer)

    def train(self):
        global_model_weights = self.training_config_helper.get_model_weights()
        personalized_model_info = self.training_config_helper.load_ditto_personalized_model_info()
        local_training_pipeline = self.training_config_helper.local_training_pipeline(global_model_weights, personalized_model_info)
        updated_global_model_weights = self.training_config_helper.return_updated_weights(local_training_pipeline)
        self.training_config_helper.save_model_global(updated_global_model_weights)
        self.training_config_helper.global_model_eval(updated_global_model_weights)
```