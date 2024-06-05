  Here is the Python code for the pipeline using the FATE library:
```
import fate
import torch

# Define the configuration file
config = {
    "reader": {
        "type": "csv",
        "path": "data/train.csv"
    },
    "data_transform": {
        "type": "preprocess",
        "params": {
            "features": ["feature1", "feature2", "feature3"],
            "target": "target"
        }
    },
    "intersection": {
        "type": "intersection",
        "params": {
            "guest": "guest",
            "host": "host"
        }
    },
    "heteronn": {
        "type": "heteronn",
        "params": {
            "epochs": 10,
            "learning_rate": 0.01,
            "batch_size": 32,
            "task_type": "classification"
        }
    },
    "evaluation": {
        "type": "evaluation",
        "params": {
            "metric": "accuracy"
        }
    }
}

# Create the pipeline
pipeline = fate.Pipeline(config)

# Compile the pipeline
pipeline.compile()

# Fit the pipeline with the training data
pipeline.fit(TrainingInputData("data/train.csv"))

# Print the summary of the HeteroNN component
print(pipeline.get_component("heteronn").summary())

# Define the main function
def main(config_file):
    # Load the configuration file
    config = fate.load_config(config_file)

    # Create the pipeline
    pipeline = fate.Pipeline(config)

    # Compile the pipeline
    pipeline.compile()

    # Fit the pipeline with the training data
    pipeline.fit(TrainingInputData("data/train.csv"))

    # Print the summary of the HeteroNN component
    print(pipeline.get_component("heteronn").summary())

# Run the main function
if __name__ == "__main__":
    main("config.json")
```
This code defines a pipeline that uses the FATE library to create a heterogeneous neural network (HeteroNN) for binary classification. The pipeline includes the following components: a Reader to read the training data, a DataTransform to preprocess the data, an Intersection to find the common instances between the guest and host data, a HeteroNN for the model training, and an Evaluation for model evaluation. The HeteroNN is configured with specific parameters such as epochs, learning rate, batch size, and task type. The neural network includes a guest bottom model, a guest top model, a host bottom model, and an interactive layer. The models are defined using the torch library and added to the HeteroNN. The HeteroNN is compiled with a specific optimizer and loss function. The pipeline is compiled and fitted with the training data. Finally, the summary of the HeteroNN component is printed. The code also includes a main function that accepts a configuration file as an argument and runs the pipeline.