  class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(self, training_config_filename, num_global_epochs, num_personalized_epochs, task_name):
        super().__init__(training_config_filename, num_global_epochs, num_personalized_epochs, task_name)
        self.helper = SupervisedPTDittoHelper(UNet, SGD)

    def train_configuration(self):
        self.helper.initialize_model()

    def train(self):
        try:
            global_weights = self.helper.get_global_model_weights()
            self.helper.prepare_fedprox_loss()
            personalized_model_info = self.helper.load_ditto_personalized_model_info()
            self.helper.local_training(global_weights, personalized_model_info)
            updated_weights = self.helper.get_updated_model_weights()
            return updated_weights
        except AbortSignal:
            return None
        except Exception as e:
            print(f"Exception occurred during training: {e}")
            return None