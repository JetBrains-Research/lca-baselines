 Here is a simplified example of how you might set up a pipeline using the FATE library for a binary classification task with a HeteroNN. Note that this is a basic example and may need to be adjusted to fit your specific needs. Also, please ensure you have the necessary installations for FATE, PyTorch, and other required libraries.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from fate_flow.api.pipeline import Pipeline
from fate_flow.api.component import Reader, DataTransform, Intersection, HeteroNN, Evaluation
from fate_flow.api.component.nn import InteractiveLayer

def define_guest_bottom(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim)
    )

def define_guest_top(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim)
    )

def define_host_bottom(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim)
    )

def define_heteronn(guest_bottom, guest_top, host_bottom, interactive_layer):
    return HeteroNN(
        guest_bottom=guest_bottom,
        guest_top=guest_top,
        host_bottom=host_bottom,
        interactive_layer=interactive_layer,
        task_type='binary_classification',
        epochs=10,
        learning_rate=0.01,
        batch_size=32,
        optimizer=optim.Adam,
        loss_function=nn.BCELoss()
    )

def main(config_file):
    # Initialize components
    reader = Reader()
    data_transform = DataTransform()
    intersection = Intersection()
    heteronn = define_heteronn(...)
    evaluation = Evaluation()

    # Configure components
    reader.configure(config_file)
    data_transform.configure(config_file)
    intersection.configure(config_file)
    heteronn.configure(config_file)

    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_component(reader)
    pipeline.add_component(data_transform)
    pipeline.add_component(intersection)
    pipeline.add_component(heteronn)
    pipeline.add_component(evaluation)

    # Fit pipeline
    pipeline.fit(...)

    # Print HeteroNN summary
    print(heteronn.summary())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_file)
```

In this example, the `define_guest_bottom`, `define_guest_top`, `define_host_bottom`, and `define_heteronn` functions define the neural network components. You should replace the ellipses (`...`) with the actual implementations of these functions based on your specific requirements. The `main` function initializes the components, configures them using a configuration file, builds the pipeline, fits the pipeline, and prints the HeteroNN summary. The script is then executed by passing the configuration file as an argument.