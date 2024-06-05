  class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(self, training_config_filename, num_aggregation_epochs, num_ditto_model_epochs, training_task_name):
        super().__init__(training_config_filename, num_aggregation_epochs, training_task_name)
        self.num_ditto_model_epochs = num_ditto_model_epochs
        self.ditto_helper = SupervisedPTDittoHelper()

    def train_config(self):
        super().train_config()
        self.ditto_helper.model = UNet(self.config.input_shape, self.config.output_shape)
        self.ditto_helper.optimizer = Adam(self.ditto_helper.model.parameters(), lr=self.config.learning_rate)

    def train(self):
        try:
            while True:
                # Handle abort signals
                if self.abort_signal:
                    break

                # Update local model weights with received weights
                self.update_local_model_weights()

                # Load Ditto personalized model
                self.ditto_helper.load_personalized_model()

                # Perform local training on the reference model and personalized model
                self.ditto_helper.train()

                # Validate the Ditto model each round
                self.ditto_helper.validate()

                # Compute the delta model
                self.ditto_helper.compute_delta_model()

                # Return a shareable object with the updated local model
                return self.ditto_helper.get_shareable_object()
        except Exception as e:
            print(f"Exception occurred in Ditto training: {e}")
            return None