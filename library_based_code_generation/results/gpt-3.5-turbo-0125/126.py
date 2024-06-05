class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(self, training_config_filename, num_aggregation_epochs, num_ditto_model_epochs, training_task_name):
        super().__init__(training_config_filename)
        self.num_aggregation_epochs = num_aggregation_epochs
        self.num_ditto_model_epochs = num_ditto_model_epochs
        self.training_task_name = training_task_name
        self.ditto_helper = SupervisedPTDittoHelper()

    def train_config(self):
        super().train_config()
        self.ditto_helper.model = UNet()
        self.ditto_helper.optimizer = Adam()

    def train(self):
        # Handle abort signals
        # Update local model weights with received weights
        # Load Ditto personalized model
        # Perform local training on the reference model and personalized model
        # Validate the Ditto model each round
        # Compute the delta model
        # Return a shareable object with the updated local model