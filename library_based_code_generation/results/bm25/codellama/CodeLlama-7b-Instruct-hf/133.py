  Here is the Python code based on the instruction:
```
import fate

# Load the configuration file
config = fate.load_config("config.json")

# Create a pipeline for a binary classification task using a HeteroNN
pipeline = fate.Pipeline(config)

# Add components for reading data, transforming data, intersecting data, and evaluating the model
pipeline.add_component(fate.ReadData())
pipeline.add_component(fate.TransformData())
pipeline.add_component(fate.IntersectData())
pipeline.add_component(fate.EvaluateModel())

# Configure the HeteroNN with specific parameters
pipeline.add_component(fate.HeteroNN(
    epochs=config["epochs"],
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    callback_params=config["callback_params"]
))

# Add a guest and host bottom model, an interactive layer, and a guest top model
pipeline.add_component(fate.GuestBottomModel())
pipeline.add_component(fate.HostBottomModel())
pipeline.add_component(fate.InteractiveLayer())
pipeline.add_component(fate.GuestTopModel())

# Compile and fit the pipeline
pipeline.compile()
pipeline.fit()

# Print the summary of the HeteroNN component
print(pipeline.summary())

# Define a main function that accepts a configuration file as an argument
def main(config_file):
    # Load the configuration file
    config = fate.load_config(config_file)

    # Create a pipeline for a binary classification task using a HeteroNN
    pipeline = fate.Pipeline(config)

    # Add components for reading data, transforming data, intersecting data, and evaluating the model
    pipeline.add_component(fate.ReadData())
    pipeline.add_component(fate.TransformData())
    pipeline.add_component(fate.IntersectData())
    pipeline.add_component(fate.EvaluateModel())

    # Configure the HeteroNN with specific parameters
    pipeline.add_component(fate.HeteroNN(
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        callback_params=config["callback_params"]
    ))

    # Add a guest and host bottom model, an interactive layer, and a guest top model
    pipeline.add_component(fate.GuestBottomModel())
    pipeline.add_component(fate.HostBottomModel())
    pipeline.add_component(fate.InteractiveLayer())
    pipeline.add_component(fate.GuestTopModel())

    # Compile and fit the pipeline
    pipeline.compile()
    pipeline.fit()

    # Print the summary of the HeteroNN component
    print(pipeline.summary())

# Call the main function with the configuration file as an argument
if __name__ == "__main__":
    main("config.json")
```