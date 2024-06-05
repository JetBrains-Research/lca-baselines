class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(self, training_config_filename, global_epochs, personalized_epochs, task_name):
        self.training_config_filename = training_config_filename
        self.global_epochs = global_epochs
        self.personalized_epochs = personalized_epochs
        self.task_name = task_name

    def training_configuration(self):
        self.helper = SupervisedPTDittoHelper(model=UNet(), optimizer=SGD())

    def training(self):
        try:
            # Get global model weights
            global_weights = self.get_global_model_weights()

            # Prepare for fedprox loss
            fedprox_loss = self.prepare_fedprox_loss()

            # Load Ditto personalized model info
            personalized_model_info = self.load_ditto_personalized_model_info()

            # Local training of reference model and personalized model
            self.local_training(global_weights, fedprox_loss, personalized_model_info)

            # Return updated weights of reference model
            return self.get_updated_weights(global_weights)

        except KeyboardInterrupt:
            print("Training aborted by user.")
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")